import emdp.gridworld  # noqa
import numpy as np
import tqdm

import environment.cube  # noqa
import models


def define_action_models(goal_state, mdp):
    action_models = []
    for a in range(mdp.num_actions):
        transition_model = mdp.transition[:, a]

        # Goal state transitions to an exiting self-looping state
        transition_model[goal_state] = mdp.transition[0, 0]

        action_model = models.Model(transition_model, mdp.reward[:, a])
        action_model.discount *= mdp.discount
        action_models.append(action_model)
    return action_models


def apmi(mdp, render=False):
    # Define Option Model
    num_states = mdp.reward.shape[0]

    value_model = np.zeros(num_states)
    goal_state = mdp.reward.max(axis=1).argmax()

    action_models = define_action_models(goal_state, mdp)

    num_iters = 100_000
    pbar = tqdm.tqdm(total=num_iters)

    for i in range(num_iters):  # a matrix-based implementation
        action_values = np.zeros((num_states, mdp.num_actions))
        for action, action_model in enumerate(action_models):
            action_value = action_model.dot(value_model)
            action_values[:, action] = action_value

        max_q = np.max(action_values, axis=1)
        improvement = max_q - value_model
        pbar.update(1)
        pbar.set_description(f"Improvement: {improvement.sum()}, nonzeros {np.count_nonzero(improvement)}")
        if render:
            _, fig = mdp.plot_s(f"Iteration: {i}", improvement)
            fig.show()
        if improvement.sum() == 0:  # or < 1e-9:
            break
        value_model[:] = max_q

    # vf = value_model
    # mdp.plot_s("vf", vf)
    # plt.show()
    return value_model


def main():
    mdp = emdp.gridworld.GridWorldMDP(goal=(1, 1), ascii_room=None)
    print("APMI for GridWorldMDP")
    apmi(mdp)

    # models.Model.factorize_transition = True
    print("APMI for CubeMDP")
    mdp = environment.cube.Cube2x2()
    apmi(mdp)


if __name__ == "__main__":
    main()
