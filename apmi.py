import matplotlib.pyplot as plt
import numpy as np
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import emdp.gridworld
import tqdm

import models


def define_action_models(empty, goal_state, mdp):
    action_models = []
    for a in range(mdp.num_actions):
        action_model = empty.copy()
        action_model.value_model = mdp.reward[:, a]

        action_model.transition_model = mdp.transition[:, a] * mdp.discount
        # Goal state transitions to an exiting self-looping state
        action_model.transition_model[goal_state, :] = mdp.transition[0, 0]

        action_models.append(action_model)
    return action_models


def main():
    mdp = emdp.gridworld.GridWorldMDP(goal=(1, 1), ascii_room=None)

    # Define Option Model
    num_states = mdp.reward.shape[0]
    empty = models.Model(num_states)

    value_model = models.Model(num_states)
    goal_state = mdp.reward.max(axis=1).argmax()

    action_models = define_action_models(empty, goal_state, mdp)

    for i in tqdm.trange(75):  # a matrix-based implementation
        action_values = np.zeros((num_states, mdp.num_actions))
        for action, action_model in enumerate(action_models):
            action_value = action_model.dot(value_model.value_model)
            action_values[:, action] = action_value

        if (value_model.value_model - np.max(action_values, axis=1)).sum() == 0.:
            break
        value_model.value_model[:] = np.max(action_values, axis=1)

    vf = value_model.value_model
    mdp.plot_s("vf", vf)
    plt.show()
    return value_model


if __name__ == "__main__":
    main()
