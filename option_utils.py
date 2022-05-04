import jax.numpy as np


def batched_option_model(mdp, option_policies, option_termination_prob):
    # n tasks, o options, transitions are s -> t

    termination = option_termination_prob
    continuation = 1 - termination
    # n = task, o = option, s = state0, t = state1, a = action

    P_pi = np.einsum('sat,osa->sot', mdp.transition, option_policies)
    r_pi = np.einsum('nsa,osa->nos', mdp.rs, option_policies)
    assert np.allclose(P_pi.sum(-1), 1)

    continuation_kernels = np.einsum('sot,ot->sot', P_pi, continuation)
    termination_kernels = np.einsum('sot,ot->sot', P_pi, termination)

    successor_features = batched_successor_features(continuation_kernels, mdp)

    reward_model = np.einsum('sot,nos->not', successor_features, r_pi)  # TODO: is it `t` or `s` on the right-hand side?
    P_model = np.einsum("sot,tok->sok", successor_features, termination_kernels)

    assert np.all(P_model > 0)
    return P_model, reward_model


def batched_successor_features(transition_kernels, mdp):
    num_states, num_actions, _num_states = transition_kernels.shape
    I = np.eye(mdp.num_states)
    successor_features = []
    for a in range(num_actions):
        action_transition_kernel = transition_kernels[:, a, :]
        successor_features.append(np.linalg.inv(I - mdp.discount * action_transition_kernel))
    successor_features = np.stack(successor_features, axis=1)
    return successor_features
