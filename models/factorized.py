import numpy as np
import time

import external.py222
import utils.enumerate_transition

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
