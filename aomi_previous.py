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
    goal_value_model_G[1 + 60, 0] = 2.
    # goal_value_model_G[1:, 0] = mdp.reward.max(axis=1)

    # Define Action Models
    action_models = []
    for a in range(mdp.num_actions):
        action_model = empty.copy()
        action_model[1:, 1:] = mdp.transition[:, a] * mdp.discount
        action_model[1:, 0] = mdp.reward[:, a]
        action_models.append(action_model)
    action_models = np.array(action_models)

    for i in range(1000):  # the most linear implementation
        if option_model_M[1:, 1:].any():
            mdp.plot_ss(f"option_model{i}", option_model_M[1:, 1:], min_weight=0.01)
            plt.savefig(f"plots/option_model{i}.png")
            plt.close()
        else:
            print("Option Model is empty at step", i)

        for s_idx in range(mdp.num_states):
            s = np.eye(mdp.num_states + 1)[s_idx + 1]
            max_val = -np.inf
            old_option_model_M = option_model_M.copy()  # save model for calculations

            for action_model in action_models:
                next_rasp_sA = s.dot(action_model)
                old_option_value_MG = np.einsum("st,tu->su", old_option_model_M, goal_value_model_G)

                continuation_rasp = next_rasp_sA.dot(old_option_value_MG)
                termination_rasp = next_rasp_sA.dot(goal_value_model_G)

                continuation_value = continuation_rasp[0]
                termination_value = termination_rasp[0]

                if termination_value > continuation_value:
                    if termination_value > max_val:
                        option_model_M[s.astype(bool)] = next_rasp_sA
                        max_val = termination_value
                else:
                    if continuation_value > max_val:
                        option_model_M[s.astype(bool)] = next_rasp_sA.dot(option_model_M)
                        max_val = continuation_value

    vf = option_model_M[1:, 0]
    mdp.plot_s(f"vf", vf)
    plt.show()
    mdp.plot_ss(f"P", option_model_M[1:, 1:])
    plt.show()


if __name__ == "__main__":
    main()
