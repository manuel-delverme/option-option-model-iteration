# Kris' enumeration of statespace
import tqdm
import time
import collections

import numpy as np

import external.py222

num_states = 3674160


def generate_state_distances_bfs():
    state_distances = np.ones(num_states, dtype=int) * 12
    index_to_stickers = np.zeros((num_states, 24))

    stickers = external.py222.initState()
    frontier = collections.deque((stickers, 0, -1))

    state_idx = external.py222.indexOP(external.py222.getOP(stickers))
    index_to_stickers[state_idx] = stickers
    state_distances[state_idx] = 0
    pbar = tqdm.tqdm(total=3674160)

    while frontier:
        stickers, depth, lm = frontier.popleft()
        for action in range(9):
            if action // 3 == lm // 3:
                continue

            next_factors = external.py222.doMove(stickers, action)
            idxp = external.py222.indexOP(external.py222.getOP(next_factors))

            if depth + 1 < state_distances[idxp]:
                state_distances[idxp] = depth + 1
                index_to_stickers[idxp] = next_factors
                pbar.update(1)
                frontier.append((next_factors, depth + 1, action))

    with open('test.npy', 'wb') as f:
        np.save(f, state_distances)
        np.save(f, index_to_stickers)


if __name__ == "__main__":
    start = time.time()
    generate_state_distances_bfs()
    print(time.time() - start)
