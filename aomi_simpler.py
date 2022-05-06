import matplotlib.pyplot as plt
import numpy as np
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
    mdp = emdp.gridworld.GridWorldMDP(goal=(1, 1), ascii_room=None)

    
    # Define Option Model
    num_states = mdp.reward.shape[0]
    empty = np.zeros((1 + num_states, 1 + num_states))
    empty[0, 0] = 1

    option_model_M = empty.copy()

    # import pdb;pdb.set_trace()
    # Define Goal Value Model
    sub_goal_state = int(num_states - np.sqrt(num_states) - 2)
    goal_value_model_G = empty.copy()
    goal_value_model_G[1 + sub_goal_state, 0] = 1.



    goal_state = np.where(mdp.reward)[0][0]

    # Define Action Models
    action_models = []
    for a in range(mdp.num_actions):
        action_model = empty.copy()
        action_model[1:, 1:] = mdp.transition[:, a] * mdp.discount
        action_model[1:, 0] = mdp.reward[:, a]

        
        # Goal state transitions to an exiting self-looping state        
        action_model[goal_state+1,1:] = mdp.transition[goal_state, 0]

        action_models.append(action_model)
    action_models = np.array(action_models)

    # import pdb;pdb.set_trace()
    for i in range(100):  # the most linear implementation
        # if i < 10 and i % 2 == 0:
        #     mdp.plot_ss(f"P{i}", option_model_M[1:, 1:], min_weight=0.01)
        #     plt.show()
        #     plt.close()

        for s_idx in range(mdp.num_states):
            s = np.eye(mdp.num_states + 1)[s_idx + 1]
            max_val = -np.inf
            old_option_model_M = option_model_M.copy()  # save model for calculations
            

            for action, action_model in enumerate(action_models):
                next_rasp_sA = s.dot(action_model)
                old_option_value_MG = np.einsum("st,tu->su", old_option_model_M, goal_value_model_G)

                continuation_rasp = next_rasp_sA.dot(old_option_value_MG)
                termination_rasp = next_rasp_sA.dot(goal_value_model_G)

                continuation_value = continuation_rasp[0]
                termination_value = termination_rasp[0]



                if termination_value >= continuation_value or s_idx==goal_state:
                    if termination_value > max_val:
                        option_model_M[s.astype(bool)] = next_rasp_sA
                        max_val = termination_value
                else:
                    if continuation_value > max_val:
                        option_model_M[s.astype(bool)] = next_rasp_sA.dot(option_model_M)
                        max_val = continuation_value


    matrix = option_model_M[1:, 1:] * (1 - np.eye(option_model_M[1:, 1:].shape[0]))

    mdp.plot_ss(f"P", matrix )
    plt.show()
    # vf = option_model_M[1:, 0]
    # mdp.plot_s("vf", vf)
    # plt.show()


if __name__ == "__main__":
    main()
