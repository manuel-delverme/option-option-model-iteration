import numpy as onp
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import option_utils


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
        import pdb;pdb.set_trace()
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
