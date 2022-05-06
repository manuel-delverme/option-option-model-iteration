import matplotlib.pyplot as plt
import numpy as np
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import emdp.gridworld


def build_action_models(mdp):
    num_states = mdp.reward.shape[0]
    empty = np.zeros((1 + num_states, 1 + num_states))
    empty[0, 0] = 1

    action_models = []
    for a in range(mdp.num_actions):
        action_model = empty.copy()
        action_model[1:, 1:] = mdp.transition[:, a] * mdp.discount
        action_model[1:, 0] = mdp.reward[:, a]
        action_models.append(action_model)
    action_models = np.array(action_models)
    return action_models


def aomi(mdp, num_state_sweeps=1000):
    # Define Option Model
    num_states = mdp.reward.shape[0]
    empty = np.zeros((1 + num_states, 1 + num_states))
    empty[0, 0] = 1

    option_model_M = empty.copy()

    # Define Goal Value Model
    goal_value_model_G = empty.copy()
    goal_value_model_G[1 + 60, 0] = 2.
    # goal_value_model_G[1:, 0] = mdp.reward.max(axis=1)

    action_models = build_action_models(mdp)

    for i in range(num_state_sweeps):  # the most linear implementation
        if option_model_M[1:, 1:].any():
            mdp.plot_ss(f"option_model{i}", option_model_M[1:, 1:], min_weight=0.01)
            plt.savefig(f"plots/option_model{i}.png")
            plt.close()
        else:
            print("Option Model is empty at step", i)

        aomi_sweep(action_models, goal_value_model_G, mdp, option_model_M)

    vf = option_model_M[1:, 0]
    mdp.plot_s(f"vf", vf)
    plt.show()
    mdp.plot_ss(f"P", option_model_M[1:, 1:])
    plt.show()
    return option_model_M


def aomi_sweep(action_models, goal_value_model_G, mdp, option_model_M):
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


def main(mdp):
    mdp = emdp.gridworld.GridWorldMDP(goal=(1, 1))
    aomi(mdp)


if __name__ == "__main__":
    main()
