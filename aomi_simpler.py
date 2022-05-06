import matplotlib.pyplot as plt
import numpy as np

import aomi
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import emdp.gridworld


def main():
    ascii_room = """
    #####
    #   #
    #   #
    #   #
    #####"""[1:].split('\n')
    mdp = emdp.gridworld.GridWorldMDP(goal=(1, 1), ascii_room=ascii_room)

    num_states = mdp.reward.shape[0]
    empty = np.zeros((1 + num_states, 1 + num_states))
    empty[0, 0] = 1

    option_model_M = empty.copy()

    # Define Goal Value Model
    goal_value_model_G = empty.copy()
    goal_value_model_G[1 + 18, 0] = 2.

    action_models = aomi.build_action_models(mdp)

    for i in range(1000):  # the most linear implementation
        if i < 10 and i % 2 == 0:
            mdp.plot_ss(f"P{i}", option_model_M[1:, 1:], min_weight=0.01)
            plt.show()
            plt.close()

        aomi.aomi_sweep(action_models, goal_value_model_G, mdp, option_model_M)

    mdp.plot_ss(f"P", option_model_M[1:, 1:])
    plt.show()
    vf = option_model_M[1:, 0]
    mdp.plot_s("vf", vf)
    plt.show()


if __name__ == "__main__":
    main()
