import tqdm
import time
import collections

import numpy as np

import external.py222

num_states = 3674160
actions_str = ["U", "U'", "F", "F'", "R", "R'"]
action_canonical = [external.py222.moveInds[a_str] for a_str in actions_str]

action_idx_to_canonical = dict(enumerate(action_canonical))
action_str_to_canonical = dict(zip(actions_str, action_canonical))
action_str_to_idx = {v: k for k, v in enumerate(actions_str)}
action_canonical_to_idx = {v: k for k, v in action_idx_to_canonical.items()}


def generate_statespace_matrices():
    factors = external.py222.initState()
    num_actions = len(action_canonical)
    # num_factors = len(factors)

    closed = np.zeros(num_states, dtype=bool)
    opened = np.zeros(num_states, dtype=bool)

    transition_function = np.full((num_states, num_actions), np.nan, dtype=np.uint)
    unhash_table = np.full((num_states, len(factors)), np.nan, dtype=np.uint)

    s0 = external.py222.indexOP(external.py222.getOP(factors))

    opened[s0] = True
    frontier = collections.deque(((factors, s0, 0, -1),))  # , maxlen=num_states)

    pbar = tqdm.tqdm(total=3674160)

    while frontier:
        factors, s0, depth, last_action = frontier.popleft()
        if closed[s0]:
            continue

        s0 = external.py222.indexOP(external.py222.getOP(factors))
        unhash_table[s0] = factors

        for action, action_encoding in action_idx_to_canonical.items():
            # if action // 3 == last_action // 3:
            #     continue

            next_factors = external.py222.doMove(factors, action_encoding)

            s1 = external.py222.indexOP(external.py222.getOP(next_factors))
            transition_function[s0, action] = s1

            if not opened[s1]:
                pbar.update(1)
                frontier.append((next_factors, s1, depth + 1, action))
                opened[s1] = True

        closed[s0] = True

    np.save("transition.npy", transition_function)
    np.save("unhash.npy", unhash_table)


if __name__ == "__main__":
    start = time.time()
    generate_statespace_matrices()
    print(time.time() - start)
