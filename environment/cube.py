import random
import time

import scipy.linalg

import utils.enumerate_transition

import emdp.gridworld
import gym
import numpy as np

import external.py222


class FactorModel:
    def __init__(self):
        self.transition = np.load("factor_transition.npy")
        num_factors, num_actions, factor_values = self.transition.shape

        self.num_factors = num_factors
        self.num_actions = num_actions
        self.factor_values = factor_values

    def __call__(self, factors, action):
        # factors = self.factorize(factors)
        # new_state = np.full_like(factors, np.nan)
        new_factors = np.full_like(factors, np.nan)

        ALL_FACTORS = np.arange(self.num_factors)
        f1 = self.transition[ALL_FACTORS, factors, action]

        for factor_idx, (old_factor_value, factor_transition) in enumerate(zip(factors, self.transition)):
            new_factors[factor_idx] = factor_transition[old_factor_value, action]

        # new_state = external.py222.doMove(factors, a_str)
        # new_state = self.defactorize(new_factors)
        return new_factors

    def factorize(self, state):
        return np.random.randint(0, self.num_factors, size=self.num_factors)

    def defactorize(self, factors):
        return np.random.randint(0, self.num_factors, size=self.num_factors)

    def factor_transition(self, factor_idx, action):
        return np.random.randint(0, self.num_factors, size=self.num_factors)


class Cube2x2:
    discount = 0.9
    metadata = {"render.modes": ["ansi", "rgb_array"]}
    actions = list(external.py222.moveInds.values())

    def render(self, mode="human"):
        factors = self._factors
        if mode == "human":
            pass
        elif mode == "normalized":
            factors = external.py222.normFC(factors)

        external.py222.printCube(factors)

    def __init__(self):
        # self.transition_models = FactorModel()
        self.transition = np.load("transition.npy")
        self.str_to_canonical = utils.enumerate_transition.action_str_to_canonical
        self.str_to_idx = utils.enumerate_transition.action_str_to_idx

    def reset(self):
        self._factors = external.py222.initState()
        self.state = self.hash(external.py222.initState())

    def step(self, action):
        # expected_f1 = external.py222.doAlgStr(self.factors, a_str)
        expected_f1 = external.py222.doMove(self._factors, self.str_to_canonical[action])
        s1 = self.transition[self.state, self.str_to_idx[action]]

        self.state = s1
        self._factors = expected_f1
        assert self.hash(expected_f1) == s1
        return s1, 0, False, {}

    def hash(self, factor_states):
        s_idx = external.py222.indexOP(external.py222.getOP(factor_states))
        return s_idx


if __name__ == "__main__":
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
