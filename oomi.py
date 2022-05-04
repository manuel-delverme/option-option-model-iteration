import numpy as onp
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import option_utils


def make_option_model(option_reward_vector, option_transition_model) -> onp.ndarray:
    num_states = option_reward_vector.shape[0]
    homogeneous_model = onp.zeros((num_states + 1, num_states + 1))
    homogeneous_model[0, 0] = 1
    homogeneous_model[1:, 1:] = option_transition_model
    homogeneous_model[1:, 0] = option_reward_vector
    return onp.array(homogeneous_model)


def make_action_model(reward_model, transition_model):
    return make_option_model(reward_model, transition_model)


def make_policy_model(value_function):
    num_states = value_function.shape[0]
    return make_option_model(value_function, onp.zeros((num_states, num_states)))


def make_identity_model(num_states, num_actions):
    return make_option_model(onp.zeros(num_states), onp.eye(num_states))


def make_value_model(value_function):
    return make_policy_model(value_function)


def eval_value_function(state, value_model) -> float:  # V(s)
    return float(onp.einsum("s,s->", value_model, state))


def make_action_value_function(state, action_model, value_model):  # Q(s, a=*)
    state_action_model = onp.einsum("s,sat->sat", state, action_model)
    action_value_function = onp.einsum("sat,sat->sat", state_action_model, value_model)
    return action_value_function


def make_option_value_function(state, action_model, option_model):  # Q(s, o=*)
    return make_action_value_function(state, action_model, option_model)


def mdp_to_value_model(mdp):
    pi_, optimal_value_function = emdp.algorithms.tabular.solve_mdp(mdp)
    return make_value_model(optimal_value_function)


def expectation_model(distribution, model):
    # TODO: does the first row matter?
    return onp.einsum("sa,sat->sat", distribution, model)


def maximizing_model(*args, **kwargs):
    raise NotImplementedError()


def action_policy_model_expectation_equation(policy, action_model, value_model):  # EQ. (8)
    """ Has fixpoint ValueModel == PolicyModel, i.e. E_pi(ActionModel) == PolicyModel """
    action_expectation_model = expectation_model(policy, action_model)
    return onp.einsum("sat,sat->sat", action_expectation_model, value_model)


def option_policy_model_expectation_equation(meta_policy, option_models):  # EQ. (11)
    """ Has fixpoint ValueModel == PolicyModel, i.e. E_pi(OptionModel) == PolicyModel """
    option_expectation_model = expectation_model(meta_policy, option_models)
    return option_expectation_model


def option_policy_model_optimality_equation(value_model, option_model, pi):  # EQ. (12)
    option_policy_model_expectation = option_policy_model_expectation_equation(pi, option_model)
    TM = onp.einsum("sat,sat->sat", option_policy_model_expectation, value_model)
    return onp.allclose(value_model, TM)


def hierarchical_policy_model_set(options):
    return options.unique(axis=1)


def make_termination_model(termination_fn, continuation_model):
    num_states, num_actions, _ = continuation_model.shape
    I = make_identity_model(num_states, num_actions)
    termination_distr = onp.einsum("s,sat->sat", termination_fn, I)
    continuation_distr = onp.einsum("s,sat->sat", 1 - termination_fn, continuation_model)
    return termination_distr + continuation_distr


def beta_option_optimization(option_models, termination_vector, G):
    OBs = [make_termination_model(termination_vector, option_model) for option_model in option_models]

    def loss_fn(old_pi):
        E_O_E_B = expectation_model(old_pi, OBs)
        OB = onp.einsum("sat,sat->", option_model, termination_model_BM)
        OG = onp.einsum("sat,sat->", OB, G)
        return OG

    initial_policy = onp.ones(option_models[0].shape) / option_models[0].shape[0]
    loss_value = loss_fn(initial_policy)

    expected_option = expectation_model(meta_policy, option_models)

    return beta_option_model


def option_option_model_composition(base_option_policies, base_option_model):
    num_states, num_actions = base_option_model.shape
    num_options = len(base_option_policies)


def option_option_model_expectation_equation(meta_policy, option_models, value_model):
    expected_option = expectation_model(meta_policy, option_models)
    return ...


def oomi(base_option_models, subgoal_models, true_value_model_Gminus, termination_vectors):
    num_tasks = len(subgoal_models)
    current_option_models = [true_value_model_Gminus.copy() for _ in range(num_tasks)]
    for k in range(100):
        for task_idx in range(num_tasks):
            old_option_model = current_option_models[task_idx]
            termination_vector = termination_vectors[task_idx]
            base_option_model = base_option_models[task_idx]

            # new_option_model = option_termination_optimization(old_option_model)
            new_option_model = beta_option_optimization(old_option_model, termination_vector, true_value_model_Gminus)  # (option_models, termination_vector, G)
            current_option_models[task_idx] = new_option_model


def main():
    mdp = emdp.chainworld.toy_mdps.dadashi_fig2d()

    num_options = 3
    mdp.rs = onp.zeros((num_options, *mdp.reward.shape))
    mdp.rs[0, 0, 0] = 1
    mdp.rs[1, 0, 1] = 1
    mdp.rs[2, 1, 0] = 1

    mdp_value_fn = mdp_to_value_model(mdp)

    option_policies = []
    option_terminations = []
    subgoal_value_models = []

    for option_idx in range(num_options):
        mdp.reward = mdp.rs[0]
        subgoal_value_models.append(mdp_to_value_model(mdp))

        pi, vf = emdp.algorithms.tabular.solve_mdp(mdp)
        pi_onehot = onp.eye(mdp.num_actions)[pi, :]
        option_policies.append(pi_onehot)
        beta = onp.random.randint(0, 2, mdp.num_states)  # TODO: where does this come from? should it be 1 at the subgoal?
        option_terminations.append(beta)

    option_policies = onp.array(option_policies)
    option_terminations = onp.array(option_terminations)
    subgoal_value_models = onp.array(subgoal_value_models)

    P_models, r_models = option_utils.batched_option_model(mdp, option_policies, option_terminations)

    option_modules = []
    for option_idx in range(num_options):
        option_reward_model = r_models[option_idx, option_idx, :]  # no cross option needed
        option_model = make_option_model(option_reward_vector=option_reward_model, option_transition_model=P_models[:, option_idx])
        option_modules.append(option_model)

    oomi(option_modules, subgoal_value_models, mdp_value_fn, option_terminations)
    print(mdp_value_fn)


if __name__ == "__main__":
    main()
