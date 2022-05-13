import random
import time

import tqdm

import utils.enumerate_transition
import numpy as np
import external.py222

try:
    unhash_table = np.load("unhash.npy").astype(int)
except FileNotFoundError:
    print("generating tables, will take ~15min...")
    start = time.time()
    utils.enumerate_transition.generate_statespace_matrices()
    print(time.time() - start)
    unhash_table = np.load("unhash.npy").astype(int)


def hash_factors(factor_states):
    s_idx = external.py222.indexOP(external.py222.getOP(factor_states))
    return s_idx


def unhash(s_idx):
    """
    you can get the separate orientation/permutation indices via %5040 and //5040,
    then orientation index is a 7-digit base-3 number and the permutation index is the lexicographic index of a permutation of 7 unique digits
    -Kris
    """
    # orientation_index, permutation_index = np.divmod(s_idx, 5040)
    # orientation_index, permutation_index = int(orientation_index), int(permutation_index)

    # permutation = [int(d) for d in np.base_repr(permutation_index, base=3)]
    # orientation = ????
    return unhash_table[s_idx]


class FactorModel:
    def __init__(self):
        f0 = external.py222.initState()
        num_factors, = f0.shape
        num_actions = len(utils.enumerate_transition.actions_str)

        self.parents_lookup = np.full((num_actions, num_factors), fill_value=-1, dtype=np.int)
        for action in range(num_actions):
            for factor in range(num_factors):
                canonical_action = utils.enumerate_transition.action_canonical[action]
                parent = external.py222.moveDefs[canonical_action, factor]
                self.parents_lookup[action, factor] = parent

    def __getitem__(self, args):
        factors, action = args
        lookup = self.parents_lookup[action]
        new_factors = factors[lookup]
        return hash_factors(new_factors)


class Cube2x2:
    def render(self, mode="human"):
        factors = self.factors
        if mode == "human":
            pass
        elif mode == "normalized":
            factors = external.py222.normFC(factors)

        external.py222.printCube(factors)

    def __init__(self):
        self.transition = FactorModel()
        self.str_to_canonical = utils.enumerate_transition.action_str_to_canonical
        self.str_to_idx = utils.enumerate_transition.action_str_to_idx
        self.state = None
        self.factors = None

    def reset(self):
        self.factors = external.py222.initState()
        self.state = hash_factors(external.py222.initState())

    def step(self, action):
        # expected_f1 = external.py222.doAlgStr(self.factors, a_str)
        expected_f1 = external.py222.doMove(self.factors, self.str_to_canonical[action])
        s1 = self.transition[self.factors, self.str_to_idx[action]]
        assert hash_factors(expected_f1) == s1

        self.state = s1
        self.factors = expected_f1
        return s1, 0, False, {}


def verify_factorization():
    env = Cube2x2()
    transition = np.load("transition.npy")
    env.reset()
    f0 = env.factors
    s0 = env.state
    f0_ = unhash(s0)
    assert (f0 == f0_).all()

    for s0 in tqdm.trange(transition.shape[0]):
        f0 = unhash(s0)
        # for a in range(transition.shape[1]):
        for a, a_str in enumerate(utils.enumerate_transition.actions_str):
            s1_factor = transition[s0, a]
            f1_factor = unhash(s1_factor)

            env.state = s0
            env.factors = f0
            s1, _, _, _ = env.step(a_str)
            assert s1_factor == env.state
            assert (f1_factor == env.factors).all()


if __name__ == "__main__":
    verify_factorization()
    env = Cube2x2()
    print(env)
    env.reset()
    env.step("U")
    possible_moves = list(utils.enumerate_transition.actions_str)
    for _ in range(100):
        a = random.choice(possible_moves)
        env.step(a)
    # do some moves
    env.render()
    env.render(mode="normalized")

    t0 = time.time()
    p_i = env.transition
    for _ in range(2):
        p_i = p_i[env.transition]
    t1 = time.time()
    print(t1 - t0)
    print(p_i.shape)
    print(p_i)
