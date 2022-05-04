import numpy as np
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import option_utils


def make_option_model(option_reward_vector, option_transition_model) -> np.ndarray:
    num_states = option_reward_vector.shape[0]
    homogeneous_model = np.zeros((num_states + 1, num_states + 1))
    homogeneous_model[0, 0] = 1
    homogeneous_model[1:, 1:] = option_transition_model
    homogeneous_model[1:, 0] = option_reward_vector
    return np.array(homogeneous_model)


def make_action_model(reward_model, transition_model):
    return make_option_model(reward_model, transition_model)


def make_policy_model(value_function):
    num_states = value_function.shape[0]
    return make_option_model(value_function, np.zeros((num_states, num_states)))


def make_identity_model(num_states, num_actions):
    return make_option_model(np.zeros(num_states), np.eye(num_states))


def make_value_model(value_function):
    return make_policy_model(value_function)


def eval_value_function(state, value_model) -> float:  # V(s)
    return float(np.einsum("s,s->", value_model, state))


def make_action_value_function(state, action_model, value_model):  # Q(s, a=*)
    state_action_model = np.einsum("s,sat->sat", state, action_model)
    action_value_function = np.einsum("sat,sat->sat", state_action_model, value_model)
    return action_value_function


def make_option_value_function(state, action_model, option_model):  # Q(s, o=*)
    return make_action_value_function(state, action_model, option_model)


def mdp_to_value_model(mdp):
    pi_, optimal_value_function = emdp.algorithms.tabular.solve_mdp(mdp)
    import pdb;pdb.set_trace()
    return make_value_model(optimal_value_function)


def expectation_model(distribution, model):
    # TODO: does the first row matter?
    return np.einsum("sa,sat->sat", distribution, model)


def maximizing_model(*args, **kwargs):
    raise NotImplementedError()


def action_policy_model_expectation_equation(policy, action_model, value_model):  # EQ. (8)
    """ Has fixpoint ValueModel == PolicyModel, i.e. E_pi(ActionModel) == PolicyModel """
    action_expectation_model = expectation_model(policy, action_model)
    return np.einsum("sat,sat->sat", action_expectation_model, value_model)


def option_policy_model_expectation_equation(meta_policy, option_models):  # EQ. (11)
    """ Has fixpoint ValueModel == PolicyModel, i.e. E_pi(OptionModel) == PolicyModel """
    option_expectation_model = expectation_model(meta_policy, option_models)
    return option_expectation_model


def option_policy_model_optimality_equation(value_model, option_model, pi):  # EQ. (12)
    option_policy_model_expectation = option_policy_model_expectation_equation(pi, option_model)
    TM = np.einsum("sat,sat->sat", option_policy_model_expectation, value_model)
    return np.allclose(value_model, TM)


def hierarchical_policy_model_set(options):
    return options.unique(axis=1)


def make_termination_model(termination_fn, continuation_model):
    num_states, num_actions, _ = continuation_model.shape
    I = make_identity_model(num_states, num_actions)
    termination_distr = np.einsum("s,sat->sat", termination_fn, I)
    continuation_distr = np.einsum("s,sat->sat", 1 - termination_fn, continuation_model)
    return termination_distr + continuation_distr


def beta_option_optimization(base_option_set, termination_fn, G):
    def loss_fn(old_pi):
        expectation_model(old_pi, base_option_set)
        termination_model_BM = make_termination_model(termination_fn, option_model)
        OB = np.einsum("sat,sat->", option_model, termination_model_BM)
        OG = np.einsum("sat,sat->", OB, G)
        return OG

    loss_value = loss_fn(current_model)

    expected_option = expectation_model(meta_policy, option_models)

    return beta_option_model


def option_option_model_composition(base_option_policies, base_option_model):
    num_states, num_actions = base_option_model.shape
    num_options = len(base_option_policies)


def option_option_model_expectation_equation(meta_policy, option_models, value_model):
    expected_option = expectation_model(meta_policy, option_models)
    return ...


def oomi(base_option_models, subgoal_models, true_value_model_Gminus):
    num_tasks = len(subgoal_models)
    current_option_models = [true_value_model_Gminus.copy() for _ in range(num_tasks)]
    for k in range(100):
        for task_idx in range(num_tasks):
            old_option_model = current_option_models[task_idx]
            base_option_model = base_option_models[task_idx]
            # new_option_model = option_termination_optimization(old_option_model)
            new_option_model = beta_option_optimization(old_option_model, base_option_model, true_value_model_Gminus)  # (base_option_set, termination_fn, G)
            current_option_models[task_idx] = new_option_model


def main():
    mdp = emdp.chainworld.toy_mdps.dadashi_fig2d()

    # Define Option Model
    option_model = np.zeros((mdp.reward.shape[0]+1,mdp.reward.shape[0]+1))
    option_model[0,0] = 1

    #Define Goal Value Model
    goal_model = np.zeros((mdp.reward.shape[0]+1,mdp.reward.shape[0]+1))
    goal_model[0,0] = 1
    goal_model[-1,0] = 2

    #Define Action Models
    action_models = []
    for a in range(mdp.num_actions):
        action_model = np.zeros((mdp.reward.shape[0]+1,mdp.reward.shape[0]+1))
        action_model[0,0] = 1
        action_model[1:,1:] = mdp.transition[a] * 0.9
        action_model[1:,0] = mdp.reward[:,a]
        action_models.append(action_model)

    # Iterate
    for i in range(100): # the most linear implementation
        for s in range(mdp.num_states):
            max_val = -np.inf
            old_option_model = option_model.copy() # save model for calculations
            for action_model in action_models:
                
                continuation_value = action_model[1+s].dot(old_option_model.dot(goal_model[:,0]))
                termination_value = action_model[1+s].dot(goal_model[:,0])


                if termination_value > continuation_value:
                    if termination_value > max_val:
                        option_model[1+s] = action_model[1+s]
                        max_val = termination_value
                else:
                    if continuation_value > max_val:
                        option_model[1+s] = action_model[1+s].dot(option_model)
                        max_val = continuation_value
    print(option_model)



    


if __name__ == "__main__":
    main()
