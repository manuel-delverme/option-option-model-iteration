import time

import utils.enumerate_transition
import numpy as np
import external.py222

try:
    unhash_table = np.load("unhash.npy").astype(int)
    transition = np.load("transition.npy")
except FileNotFoundError:
    print("generating tables, will take ~15min...")
    start = time.time()
    utils.enumerate_transition.generate_statespace_matrices()
    print(time.time() - start)
    unhash_table = np.load("unhash.npy").astype(int)

f0 = external.py222.initState()
num_factors, = f0.shape
num_actions = len(utils.enumerate_transition.actions_str)

one_step_parents_lookup = np.full((num_actions, num_factors), fill_value=-1, dtype=np.int)
for action in range(num_actions):
    for factor in range(num_factors):
        canonical_action = utils.enumerate_transition.action_canonical[action]
        parent = external.py222.moveDefs[canonical_action, factor]
        one_step_parents_lookup[action, factor] = parent


def hash_factors(factor_states):
    s_idx = external.py222.indexOP(external.py222.getOP(factor_states))
    return s_idx


class Model:
    factorize_transition = False

    def __init__(self, transition_model, value_model):
        self.discount = 1.
        num_states = value_model.shape[0]

        if self.factorize_transition:
            self._transition_model = FactoredTransitionModel(num_states)
        else:
            if len(transition_model.shape) == 2:
                transition_model = transition_model.argmax(axis=1)
            self._transition_model = transition_model
        self._value_model = value_model
        self.num_states = num_states

    @property
    def value_model(self):
        return self._value_model

    @property
    def transition_model(self):
        return self._transition_model

    @transition_model.setter
    def transition_model(self, value):
        if isinstance(self._transition_model, FactoredTransitionModel):
            raise NotImplementedError
        else:
            self._transition_model = value

    def copy(self):
        return self.__class__(self.num_states)

    def __getitem__(self, args):
        return self.model[args]

    def dot(self, other):
        if len(other.shape) == 1:
            return other[self.transition_model] * self.discount + self.value_model
        else:
            raise NotImplementedError

    # def __setitem__(self, slice_, value):
    #     if slice_ == self.P:
    #         self.transition_model[:] = value
    #     elif slice_ == self.R:
    #         self.value_model[:] = value
    #     else:
    #         slice1, slice2 = slice_
    #         if not isinstance(slice1, slice):
    #             slice1 = slice(slice1, slice1 + 1)
    #         if not isinstance(slice2, slice):
    #             slice2 = slice(slice2, slice2 + 1)

    #         if slice1.start > 0 and slice2.start > 0:
    #             slice1 = slice(slice1.start - 1, slice1.stop - 1, slice1.step)
    #             slice2 = slice(slice2.start - 1, slice2.stop - 1, slice2.step)
    #             self.transition_model[slice1, slice2] = value
    #         elif slice1.start == 0:
    #             if slice2.start == 0:
    #                 raise ValueError
    #             self.value_model[slice1, slice2] = value
    #         elif slice2.start == 0:
    #             self.value_model[slice1, slice2] = value
    #         else:
    #             raise ValueError

    #         raise NotImplementedError


class FactoredTransitionModel:
    def __init__(self, num_states):
        raise NotImplementedError("it should be hardcoded for cube factors, for simplcitiy")
        self.num_states = num_states

    def __getitem__(self, args):
        factor_values, action = args
        parent_factor = one_step_parents_lookup[action]
        new_factors = factor_values[parent_factor]
        return hash_factors(new_factors)

    def __setitem__(self, key, value):
        raise NotImplementedError
