import matplotlib.pyplot as plt
import numpy as np
import emdp.chainworld.toy_mdps
import emdp.algorithms.tabular
import emdp.gridworld


def main():



    # ascii_room = """
    # #####
    # #   #
    # #   #
    # #   #
    # #####"""[1:].split('\n')

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


    # ascii_room = """
    # #####################################
    # #   #   #   #   #   #   #   #   #   #
    # #           #           #           #
    # #   #   #   #   #   #   #   #   #   #
    # ## ### ### ### ### ### ### ### ### ##
    # #   #   #       #   #       #   #   #
    # #                                   #
    # #   #   #       #   #       #   #   #
    # ## ### ### ### ### ### ### ### ### ##
    # #   #   #   #   #   #   #   #   #   #
    # #           #           #           #
    # #   #   #   #   #   #   #   #   #   #
    # #####   #########   #########   #####
    # #   #   #   #   #   #   #   #   #   #
    # #           #           #           #
    # #   #   #   #   #   #   #   #   #   #
    # ## ### ### ### ### ### ### ### ### ##
    # #   #   #       #   #       #   #   #
    # #                                   #
    # #   #   #       #   #       #   #   #
    # ## ### ### ### ### ### ### ### ### ##
    # #   #   #   #   #   #   #   #   #   #
    # #           #           #           #
    # #   #   #   #   #   #   #   #   #   #
    # #####   #########   #########   #####
    # #   #   #   #   #   #   #   #   #   #
    # #           #           #           #
    # #   #   #   #   #   #   #   #   #   #
    # ## ### ### ### ### ### ### ### ### ##
    # #   #   #       #   #       #   #   #
    # #                                   #
    # #   #   #       #   #       #   #   #
    # ## ### ### ### ### ### ### ### ### ##
    # #   #   #   #   #   #   #   #   #   #
    # #           #           #           #
    # #   #   #   #   #   #   #   #   #   #
    # #####################################"""[1:].split('\n')
    mdp = emdp.gridworld.GridWorldMDP(goal=(1, 1), ascii_room=None)
    print("MDP ready.")
    
    # Define Option Model
    num_states = mdp.reward.shape[0]
    empty = np.zeros((1 + num_states, 1 + num_states))
    empty[0, 0] = 1

    value_model = empty.copy()
    goal_state = np.where(mdp.reward)[0][0]

    # Define Action Models
    action_models = []
    for a in range(mdp.num_actions):
        action_model = empty.copy()
        action_model[1:, 1:] = mdp.transition[:, a] * mdp.discount
        action_model[1:, 0] = mdp.reward[:, a]
        # Goal state transitions to an exiting self-looping state        
        action_model[1+goal_state,1:] = mdp.transition[0, 0] 
        action_models.append(action_model)
    action_models = np.array(action_models)

    
    # for i in range(10):  # the most linear implementation

    #     # old_value_model = np.copy(value_model)
    #     # import pdb;pdb.set_trace()
    #     for s_idx in range(mdp.num_states):
    #         print(s_idx)
    #         s = np.eye(mdp.num_states + 1)[s_idx + 1]
    #         max_val = -np.inf
            
    #         old_value_model = np.copy(value_model)

    #         for action, action_model in enumerate(action_models):
    #             next_rasp_sA = s.dot(action_model)
    #             action_value_row = next_rasp_sA.dot(old_value_model)

    #             if action_value_row[0] > max_val:
    #                 value_model[s_idx + 1] = action_value_row
    #                 max_val = action_value_row[0]


    for i in range(75):  # a matrix-based implementation

        action_values=np.zeros((num_states+1,mdp.num_actions))
        for action, action_model in enumerate(action_models):
            action_value = action_model.dot(value_model[:,0])
            action_values[:,action] = action_value

        # import pdb;pdb.set_trace()
        print(i,(value_model[:,0] - np.max(action_values,axis=1)).sum())
        if (value_model - np.max(action_values,axis=1)).sum() == 0.:
            print(i)
            break
        value_model[:,0] = np.max(action_values,axis=1)


    vf = value_model[1:, 0]
    mdp.plot_s("vf", vf)
    plt.show()
    return value_model


if __name__ == "__main__":
    main()
