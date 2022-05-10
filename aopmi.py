import matplotlib.pyplot as plt
import numpy as np
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import emdp.gridworld


def intra_option_learning(mdp, action_models, true_value_model, sub_goal_state, sub_goal_initiation_set):
    # import pdb;pdb.set_trace()

    num_states = mdp.reward.shape[0]
    empty = np.zeros((1 + num_states, 1 + num_states))
    empty[0, 0] = 1

    goal_state = np.where(mdp.reward)[0][0]
    # option_model_M = true_value_model.copy()
    option_model_M = empty.copy()

    # Define Goal Value Model
    goal_value_model_G = empty.copy()
    goal_value_model_G[1 + sub_goal_state, 0] = 1 if sub_goal_state != goal_state else 0.


    value_model=empty.copy()

    for i in range(5): # 4 Iterations per option (first one is artefact of MDP)

        old_option_model_M = option_model_M.copy()  # save model for calculations
        old_value_model = np.copy(value_model)


        for s_idx in sub_goal_initiation_set:
            s = np.eye(mdp.num_states + 1)[s_idx + 1]
            max_val = -np.inf

            # old_option_model_M = option_model_M.copy()  # save model for calculations

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
                        value_model[1+s_idx] = termination_rasp
                        max_val = termination_value
                else:
                    if continuation_value > max_val:
                        option_model_M[s.astype(bool)] = next_rasp_sA.dot(option_model_M)
                        value_model[1+s_idx] = continuation_value
                        max_val = continuation_value


    # import pdb;pdb.set_trace()
        # matrix = option_model_M[1:, 1:] * (1 - np.eye(option_model_M[1:, 1:].shape[0]))
        # mdp.plot_ss(f"P", matrix, min_weight=0.)
        # plt.show()

    # vf = value_model[1:, 0]
    # mdp.plot_s(f"vf_subgoal{sub_goal_state}", vf)
    # plt.show()

    return option_model_M

def main():

    # from apmi import main as apmi
    # true_value_model = apmi()
    true_value_model=None

    ascii_room = """
    #############
    #   #   #   #
    #           #
    #   #   #   #
    ## ### ### ##
    #   #   #   #
    #           #
    #   #   #   #
    ## ### ### ##
    #   #   #   #
    #           #
    #   #   #   #
    #############"""[1:].split('\n')

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


    ### Define Action Models
    num_states = mdp.reward.shape[0]
    empty = np.zeros((1 + num_states, 1 + num_states))
    empty[0, 0] = 1
    action_models = []
    for a in range(mdp.num_actions):
        action_model = empty.copy()
        action_model[1:, 1:] = mdp.transition[:, a] * mdp.discount
        action_model[1:, 0] = mdp.reward[:, a]
        # Goal state transitions to an exiting self-looping state        
        action_model[goal_state+1,1:] = mdp.transition[0, 0] 
        action_models.append(action_model)
    action_models = np.array(action_models)
    ###


    ## Learn options, this process counts for 4*2 iterations
    # subgoals = [22,38,42,58,goal_state]
    subgoals = [22,38,58,goal_state]

    subgoal_initiationset={22:[14,15,16,23,24,25,32,33,34,42],
    38:[46,47,48,55,56,57,64,65,66,58,    50,51,52,59,60,61,68,69,70 ],
    # 42:[50,51,52,59,60,61,68,69,70],
    58:[50,51,52,59,60,61,68,69,70],
    goal_state:[10,11,12,19,20,21,28,29,30,22,38],}


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

    option_models = []
    for subgoal in subgoals:
        option_model_M = intra_option_learning(mdp, action_models, true_value_model, subgoal, subgoal_initiationset[subgoal])
        option_models.append(option_model_M)
    ###

    
    # import pdb;pdb.set_trace()
    option_models.extend(action_models)

    ### SMDP Planning
    value_model = empty.copy()
    for i in range(6):  
        
        old_value_model = np.copy(value_model)

        for s_idx in range(mdp.num_states):
            s = np.eye(mdp.num_states + 1)[s_idx + 1]
            max_val = -np.inf
            
            # old_value_model = np.copy(value_model)

            for option, option_model in enumerate(option_models):
                
                next_rasp_sO = s.dot(option_model)
                option_value_row = next_rasp_sO.dot(old_value_model)

                # if option==4 and s_idx==22:
                
                if option_value_row[0] > max_val:
                    value_model[s_idx + 1] = option_value_row
                    max_val = option_value_row[0]



        vf = value_model[1:, 0]
        mdp.plot_s("vf", vf)
        plt.show()

    # print(np.sum(value_model-true_value_model))

if __name__ == "__main__":
    main()
