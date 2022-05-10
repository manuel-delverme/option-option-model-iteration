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

    mdp = emdp.gridworld.GridWorldMDP(goal=(1, 1), ascii_room=None)

    
    # Define Option Model
    num_states = mdp.reward.shape[0]
    empty = np.zeros((1 + num_states, 1 + num_states))
    empty[0, 0] = 1

    option_model_M = empty.copy()


    # Define Goal Value Model
    sub_goal_state = int(num_states - np.sqrt(num_states) - 2)
    # sub_goal_state = 114
    goal_value_model_G = empty.copy()
    goal_value_model_G[1 + sub_goal_state, 0] = 1.2



    goal_state = np.where(mdp.reward)[0][0]

    ### Define Action Models
    action_models = []
    for a in range(mdp.num_actions):
        
        action_model = empty.copy()
        action_model[1:, 1:] = mdp.transition[:, a] * mdp.discount
        action_model[1:, 0] = mdp.reward[:, a]
        # Goal state transitions to an exiting self-looping state        
        action_model[goal_state+1,1:] = mdp.transition[0, 0] 
        action_models.append(action_model)
    # action_models = np.array(action_models)

    # import pdb;pdb.set_trace()
    all_models = action_models
    all_models.append(option_model_M)
    all_models = np.array(all_models)

    
    for i in range(2):  # the most linear implementation
        # import pdb;pdb.set_trace()
        old_option_model_M = option_model_M.copy()  # save model for calculations
        all_models[-1] = option_model_M
        
        for s_idx in range(mdp.num_states):
            s = np.eye(mdp.num_states + 1)[s_idx + 1]
            max_val = -np.inf
            
            # old_option_model_M = option_model_M.copy()  # save model for calculations


            for idx, model in enumerate(all_models):
                # if i==0 and s_idx==16:
                #     import pdb;pdb.set_trace()
                # if i >=6 and s_idx==17:
                    # if idx == len(all_models)-1:
                    #     import pdb;pdb.set_trace()
                next_rasp_sA = s.dot(model)
                old_option_value_MG = np.einsum("st,tu->su", old_option_model_M, goal_value_model_G)

                continuation_rasp = next_rasp_sA.dot(old_option_value_MG)
                termination_rasp = next_rasp_sA.dot(goal_value_model_G)

                continuation_value = continuation_rasp[0]
                termination_value = termination_rasp[0]


                if termination_value >= continuation_value or 1 in np.where(next_rasp_sA)[0]: # Force termination if reaching exiting state
                    if termination_value > max_val:
                        option_model_M[s.astype(bool)] = next_rasp_sA
                        max_val = termination_value
                else:
                    if continuation_value > max_val:
                        option_model_M[s.astype(bool)] = next_rasp_sA.dot(option_model_M)
                        max_val = continuation_value


        matrix = option_model_M[1:, 1:] * (1 - np.eye(option_model_M[1:, 1:].shape[0]))
        mdp.plot_ss(f"P_{i}", matrix )
        plt.show()
        # vf = option_model_M[1:, 0]
        # mdp.plot_s("vf", vf)
        # plt.show()


if __name__ == "__main__":
    main()
