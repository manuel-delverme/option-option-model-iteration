import copy

import numpy as np

import external.py222
import utils.enumerate_transition

# try:
#     unhash_table = np.load("unhash.npy").astype(int)
#     transition = np.load("transition.npy")
# except FileNotFoundError:
#     print("generating tables, will take ~15min...")
#     start = time.time()
#     utils.enumerate_transition.generate_statespace_matrices()
#     print(time.time() - start)
#     unhash_table = np.load("unhash.npy").astype(int)
#
# f0 = external.py222.initState()
# num_factors, = f0.shape
# num_actions = len(utils.enumerate_transition.actions_str)

# one_step_parents_lookup = np.full((num_actions, num_factors), fill_value=-1, dtype=np.int)
# for action in range(num_actions):
#     for factor in range(num_factors):
#         canonical_action = utils.enumerate_transition.action_canonical[action]
#         parent = external.py222.moveDefs[canonical_action, factor]
#         one_step_parents_lookup[action, factor] = parent


# def hash_factors(factor_states):
#     s_idx = external.py222.indexOP(external.py222.getOP(factor_states))
#     return s_idx


class DeterministicModel:
    def __init__(self, transition_matrix, value_model, discount):
        self.discount = discount
        self.void_state = transition_matrix.shape[0]

        if len(transition_matrix.shape) == 2:
            void_transitions = transition_matrix.max(axis=1) == 0
            transition_matrix = transition_matrix.argmax(axis=1)
            transition_matrix[void_transitions] = self.void_state

        self.transition_model = transition_matrix
        self._value_model = value_model

    @property
    def value_model(self):
        return self._value_model * self.discount

    @value_model.setter
    def value_model(self, value):
        self._value_model = value

    def copy(self):
        return DeterministicModel(copy.deepcopy(self.transition_model), copy.deepcopy(self.value_model), self.discount)

    def compose(self, other):
        if isinstance(other, DeterministicModel):
            void_transition = self.transition_model == self.void_state

            P1 = np.full_like(self.transition_model, fill_value=self.void_state)
            R1 = np.zeros_like(self.transition_model)

            s1 = self.transition_model[~void_transition]
            P1[~void_transition] = other.transition_model[s1]
            R1[~void_transition] = other.value_model[s1]
            return DeterministicModel(P1, R1, self.discount * other.discount)

        elif isinstance(other, np.ndarray):
            if len(other.shape) == 1:
                if isinstance(self.transition_model, np.ndarray):
                    return other[self.transition_model] * self.discount + self.value_model
                else:
                    return self.transition_model(other) * self.discount + self.value_model
        else:
            raise NotImplementedError

    def project(self, state_idx, value):
        s1 = self.transition_model[state_idx]
        return s1, value

    def __getitem__(self, state_idx):
        if state_idx == self.void_state:
            return self.void_state, 0
        return self.transition_model[state_idx], self.value_model[state_idx]

    def __setitem__(self, state_idx, value_):
        new_state, new_value = value_
        self.transition_model[state_idx] = new_state
        self.value_model[state_idx] = new_value
class FactoredTransitionModel:
    def __init__(self, num_states):
        if num_states != unhash_table.shape[0]:
            raise ValueError("num_states does not match the factor lookup table")
        self.batch_size = 1024
        self.num_states = num_states
        # raise NotImplementedError("it should be hardcoded for cube factors, for simplcitiy")
        # self.num_states = num_states

    def __getitem__(self, args):
        factor_values, action = args
        parent_factor = one_step_parents_lookup[action]
        new_factors = factor_values[parent_factor]
        return hash_factors(new_factors)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __call__(self, values):
        states = np.arange(self.num_states)
        factors = unhash_table[states]

        for batch in range(0, values.shape[0], self.batch_size):
            value_batch = values[batch:batch + self.batch_size]

            for action in range(values.shape[1]):
                parent_factor = one_step_parents_lookup[action]
                new_factors = values[batch, action, parent_factor]
                values[batch, action] = hash_factors(new_factors)
        return self[values]
