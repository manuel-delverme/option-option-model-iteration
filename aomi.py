import matplotlib.pyplot as plt
import numpy as np
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import emdp.gridworld


def main():
    mdp = emdp.gridworld.GridWorldMDP(goal=(1, 1))

    # Define Option Model
    num_states = mdp.reward.shape[0]
    empty = np.zeros((1 + num_states, 1 + num_states))
    empty[0, 0] = 1

    option_model_M = empty.copy()

    # Define Goal Value Model
    goal_value_model_G = empty.copy()
    bottleneck_value = np.zeros(num_states)
    # bottleneck_value[[22, 38, 42, 58]] = 1 # Bottlenecks
    goal_value_model_G[1:, 0] = bottleneck_value  # mdp.reward.max(axis=1)

    # Define Action Models
    action_models = []
    for a in range(mdp.num_actions):
        action_model = empty.copy()
        action_model[1:, 1:] = mdp.transition[:, a] * mdp.discount
        action_model[1:, 0] = mdp.reward[:, a]
        action_models.append(action_model)
    action_models = np.array(action_models)
    no_op_model = np.eye(1 + num_states)

    for i in range(100):  # the most linear implementation
        for s_idx in range(mdp.num_states):
            s = np.eye(mdp.num_states + 1)[s_idx + 1]
            max_val = -np.inf
            old_option_model_M = option_model_M.copy()  # save model for calculations
            continuation_model = np.einsum("st,tu->su", old_option_model_M, goal_value_model_G)
            termination_model = np.einsum("st,tu->su", no_op_model, goal_value_model_G)

            for action_model in action_models:
                next_rasp_sA = s.dot(action_model)

                continuation_rasp = next_rasp_sA.dot(continuation_model)
                termination_rasp = next_rasp_sA.dot(termination_model)

                continuation_value = continuation_rasp[0]
                termination_value = termination_rasp[0]

                if termination_value > continuation_value:
                    if termination_value > max_val:
                        new_best_rasp = next_rasp_sA
                        max_val = termination_value
                else:
                    if continuation_value > max_val:
                        new_best_rasp = next_rasp_sA.dot(option_model_M)
                        max_val = continuation_value

            option_model_M[s.astype(bool)] = new_best_rasp

    vf = option_model_M[1:, 0]
    mdp.plot_s("vf", vf)
    plt.show()


if __name__ == "__main__":
    main()
