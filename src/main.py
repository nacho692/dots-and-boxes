import sys
import numpy as np
import random
import pickle
import os.path

from learning_player import BoardSaver
from dots_boxes import DotsAndBoxes, DotsAndBoxesMaxIfKnownPolicy, DotsAndBoxesRandomPolicy, DotsAndBoxesState

random.seed(0)


def epsilon_greedy(Q, board: DotsAndBoxesState | None, action_spaces, epsilon):
    if board is None:
        return None

    max_action = max(action_spaces, key=lambda a: Q.get(board, a))
    probs = list(
        map(
            lambda x: 1 - epsilon + epsilon / len(action_spaces) if max_action == x else epsilon / len(action_spaces),
            action_spaces,
        )
    )
    return random.choices(list(action_spaces), probs).pop()


def q_learning(
    env: DotsAndBoxes,
    num_episodes: int,
    alpha: float,
    gamma: float = 1.0,
    eps: float = 1.0,
    eps_decay: float = 0.9999,
    epsmin: float = 0.01,
    Q: BoardSaver = None,
):
    if Q is None:
        Q = BoardSaver(env.size)
    rw = 0
    new_states = 0
    won = 0

    for e in range(num_episodes):
        state = env.reset()
        if not Q.contains(state):
            for a in env.action_spaces:
                Q.define(state, a, np.random.rand())
            new_states += 1

        action = epsilon_greedy(Q, state, env.action_spaces, eps)
        done = False
        while not done:
            next_state, info = env.step(action)
            reward = info.get("reward")
            done = info.get("done")
            rw += reward

            if done:
                next_state = None
                won += info.get("player_1_points") > info.get("player_2_points")

            elif not Q.contains(next_state):
                for a in env.action_spaces:
                    Q.define(next_state, a, np.random.rand())
                new_states += 1

            next_action = epsilon_greedy(Q, next_state, env.action_spaces, eps)

            eps = max(epsmin, eps * eps_decay)

            old_q_value = Q.get(state, action)
            next_expected_value = (
                max(map(lambda a: Q.get(next_state, a), env.action_spaces)) if next_state is not None else 0
            )
            new_q_value = old_q_value + alpha * (reward + gamma * next_expected_value - old_q_value)
            Q.define(state, action, new_q_value)

            state = next_state
            action = next_action
        if e % 100 == 0:
            avg_rw = rw / 100
            avg_won = won / 100
            print(
                "episode: {}, reward rate: {}, new states: {}, epsilon: {}, avg_won: {}".format(
                    e, avg_rw, new_states, eps, avg_won
                )
            )
            new_states = 0
            rw = 0
            won = 0

    return Q


board_size = 2
q_file = f"q_value_function_{board_size}x{board_size}.pickle"

if not os.path.exists(q_file):
    with open(q_file, "wb") as handle:
        pickle.dump(BoardSaver(board_size), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(q_file, "rb") as handle:
    q_value_function = pickle.load(handle)

print(len(q_value_function.boards))

for e in range(1):
    print(e)
    with open(q_file, "rb") as handle:
        training_q_value_function = pickle.load(handle)

    env = DotsAndBoxes(board_size, DotsAndBoxesRandomPolicy(training_q_value_function))
    q_value_function = q_learning(
        env, 50_000, alpha=0.05, gamma=0.95, eps=0.1, epsmin=0.01, eps_decay=0.999995, Q=q_value_function
    )
    del training_q_value_function
    with open(q_file, "wb") as handle:
        pickle.dump(q_value_function, handle, protocol=pickle.HIGHEST_PROTOCOL)

# inp = ""
# env = DotsAndBoxes(board_size, DotsAndBoxesMaxIfKnownPolicy(q_value_function))
# env.render()
# while inp != "quit":
#     inp = sys.stdin.readline()

#     def splitter(n):
#         node = list(map(int, n.split(",")))
#         return node[0], node[1]

#     action = list(map(splitter, inp.split(" ")))
#     observation, info = env.step(action)
#     done = info.get("done")

#     env.render()
#     if done:
#         env.reset()
#         env.render()
