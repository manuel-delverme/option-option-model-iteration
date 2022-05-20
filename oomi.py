import matplotlib.pyplot as plt
import numpy as np
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import emdp.gridworld
import apmi
import constants
import models


def intra_option_learning(mdp, action_models, sub_goal_state, sub_goal_initiation_set):
    num_states = mdp.num_states
    goal_state = np.where(mdp.reward)[0][0]
    option_model_M = models.DeterministicModel(np.zeros((num_states, num_states)), np.zeros(num_states), discount=mdp.discount)

    # Define Goal Value Model
    goal_value_model_G_ = np.zeros((num_states, 1))
    goal_value_model_G_[sub_goal_state, 0] = 1 if sub_goal_state != goal_state else 0.

    goal_value_model_G = models.DeterministicModel(
        np.zeros((num_states, num_states)),
        goal_value_model_G_[1:, 0],
        discount=1.
    )
    old_P = None

    for i in range(100):  # 4 Iterations per option (first one is artefact of MDP)
        old_option_model_M = option_model_M.copy()  # save model for calculations
        old_option_value_MG = old_option_model_M.compose(goal_value_model_G)

        for s_idx in sub_goal_initiation_set:
            max_val = -np.inf

            for action, action_model in enumerate(action_models):
                s1, r = action_model[s_idx]
                cont_s2, continuation_value = old_option_value_MG[s1]
                term_s2, termination_value = goal_value_model_G[s1]

                if termination_value >= continuation_value or s_idx == goal_state:
                    if termination_value > max_val:
                        option_model_M[s_idx] = (s1, r)
                        max_val = termination_value
                else:
                    if continuation_value > max_val:
                        option_model_M[s_idx] = option_model_M.project(s1, r)
                        max_val = continuation_value

        P = option_model_M.transition_model.copy()
        if i > 0:
            no_edge = P == option_model_M.void_state
            improvement = (P != old_P)[~no_edge].sum()
            print("Iteration:", i, "Improvement:", improvement)
            if improvement == 0:
                break
        old_P = P
        plot_model(mdp, option_model_M)

    return option_model_M


def plot_model(mdp, option_model_M):
    P = option_model_M.transition_model.copy()
    no_edge = P == option_model_M.void_state
    P[no_edge] = 0.
    # P = P.astype(int)
    P_matrix = np.eye(P.shape[0])[P]
    P_matrix[no_edge] = 0.
    matrix = P_matrix * (1 - np.eye(P_matrix.shape[0]))  # What happens here?
    matrix[no_edge] += (np.eye(P_matrix.shape[0]) * 0.01)[no_edge]
    mdp.plot_ss(f"P", matrix, min_weight=0.)
    plt.show()


def main():
    true_value_model = None
    ascii_room = """
    #########
    #   #   #
    #       #
    #   #   #
    ## ######
    #   #   #
    #       #
    #   #   #
    #########"""[1:].split('\n')

    mdp = emdp.gridworld.GridWorldMDP(goal=(1, 1), ascii_room=ascii_room)
    goal_state = np.where(mdp.reward)[0][0]

    num_states = mdp.reward.shape[0]
    # empty = models.Model(num_states)
    # empty = np.zeros((1 + num_states, 1 + num_states))
    # empty[0, 0] = 1
    # action_models = []
    action_models = apmi.define_action_models(goal_state, mdp)

    # Learn options, this process counts for 4*2 iterations
    # subgoals = [22,38,42,58,goal_state]
    subgoals = [
        constants.NORTH_BOTTLENECK,
        constants.SOUTH_BOTTLENECK,
        constants.WEST_BOTTLENECK,
        goal_state
    ]

    subgoal_initiationset = {
        # constants.NORTH_BOTTLENECK: [
        #     12,     14, 15, 16,
        #     21,     23, 24, 25,
        #     30,     32, 33, 34,
        # ],
        constants.NORTH_BOTTLENECK: [
            14, 15, 16,
            23, 24, 25,
            32, 33, 34,
        ],
        constants.WEST_BOTTLENECK: [46, 47, 48, 55, 56, 57, 64, 65, 66, 58, 50, 51, 52, 59, 60, 61, 68, 69, 70],
        # EAST_BOTTLENECK:[50,51,52,59,60,61,68,69,70],
        constants.SOUTH_BOTTLENECK: [50, 51, 52, 59, 60, 61, 68, 69, 70],
        goal_state: [10, 11, 12, 19, 20, 21, 28, 29, 30, 22, 38],
    }

    # subgoals=[goal_state,30,34,54,58,62,106,110,114]
    # subgoal_initiationset={goal_state:[14,15,16,27,28,29,40,41,42,54,30],
    # 30:[18,19,20,31,32,33,44,45,46,34,58],
    # 34:[22,23,24,35,36,37,48,49,50,62],
    # 54:[66,67,68,79,80,81,92,93,94,106,82],
    # 58:[70,71,72,83,84,85,96,97,98,73,97,86,110],
    # 62:[74,75,76,87,88,89,100,101,102,114],
    # 106:[118,119,120,131,132,133,144,145,146,134],
    # 110:[122,123,124,135,136,137,148,149,150,138],
    # 114:[126,127,128,139,140,141,152,153,154],}

    # subgoal_initiationset={goal_state:[14,15,16,27,28,29,40,41,42,54,30],
    # 30:[18,19,20,31,32,33,44,45,46,58],
    # 34:[22,23,24,35,36,37,48,49,50,62],
    # 54:[66,67,68,79,80,81,92,93,94,106],
    # 58:[70,71,72,83,84,85,96,97,98,73,97,110],
    # 62:[74,75,76,87,88,89,100,101,102,114],
    # 106:[118,119,120,131,132,133,144,145,146,134],
    # 110:[122,123,124,135,136,137,148,149,150,138],
    # 114:[126,127,128,139,140,141,152,153,154],}

    option_models = action_models
    for subgoal in subgoals:
        init_set = subgoal_initiationset[subgoal]
        option_model_M = intra_option_learning(mdp, option_models, subgoal, init_set)
        option_models.append(option_model_M)

    ### SMDP Planning
    value_model = models.DeterministicModel(num_states)
    for i in range(2):

        old_value_model = np.copy(value_model)

        for s_idx in range(mdp.num_states):
            s = np.eye(mdp.num_states + 1)[s_idx + 1]
            max_val = -np.inf

            # old_value_model = np.copy(value_model)

            for option, option_model in enumerate(option_models):

                next_rasp_sO = s.compose(option_model)
                option_value_row = next_rasp_sO.compose(old_value_model)

                # if option==4 and s_idx==22:

                if option_value_row[0] > max_val:
                    value_model[s_idx + 1] = option_value_row
                    max_val = option_value_row[0]

        vf = value_model[1:, 0]
        mdp.plot_s("vf", vf)
        plt.show()

    # print(np.sum(value_model-true_value_model))


if __name__ == "__main__":
    # TODO: http://algdb.net/puzzle/222
    # 1) implement ^
    #     do OOMI
    main()
