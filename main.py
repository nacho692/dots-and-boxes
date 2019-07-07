import sys
from learning_player import *
from dots_boxes import DotsAndBoxes, DotsAndBoxesRandomPolicy, DotsAndBoxesMaxIfKnownPolicy
import numpy as np
import random
import pickle
import os.path
import time
random.seed(0)

def epsilon_greedy(Q, board, action_spaces, epsilon):
    if board is None:
        return None

    max_action = max(action_spaces, key=lambda a: Q.get(board, a))
    probs = list(
        map(lambda x: 1 - epsilon + epsilon / len(action_spaces)
            if max_action == x else epsilon / len(action_spaces),
            action_spaces))
    return random.choices(list(action_spaces), probs).pop()


def q_learning(env, num_episodes, alpha, gamma=1.0, eps=1.0, eps_decay=.9999, epsmin=0.01, Q=None):
    if Q is None:
        Q = BoardSaver(env.size + 1)
    rw = 0
    new_boards = 0
    over_1k = 3
    for e in range(num_episodes):
        board = env.reset()
        if not Q.contains(board):
            for a in env.action_spaces:
                Q.define(board, a, np.random.rand())
            new_boards += 1

        action = epsilon_greedy(Q, board, env.action_spaces, eps)
        done = False
        while not done:
            next_board, reward, done, info = env.step(action)
            rw += reward

            if done:
                next_board = None
            elif not Q.contains(next_board):
                for a in env.action_spaces:
                    Q.define(next_board, a, np.random.rand())
                new_boards += 1

            next_action = epsilon_greedy(Q, next_board, env.action_spaces, eps)

            eps = max(epsmin, eps * eps_decay)

            old_q_value = Q.get(board, action)
            next_expected_value = max(
                map(lambda a: Q.get(next_board, a), env.action_spaces)) if next_board is not None else 0
            new_q_value = old_q_value + \
                          alpha*(reward + gamma*next_expected_value - old_q_value)
            Q.define(board, action, new_q_value)

            board = next_board
            action = next_action
        if e % 100 == 0:
            avg_rw = rw/100
            print("episode: {}, reward rate: {}, new boards: {}, epsilon: {}, over_1k: {}"
                  .format(e, avg_rw, new_boards, eps, over_1k))
            new_boards = 0
            rw = 0
            if avg_rw > 1:
                over_1k -= 1
                if over_1k == 0:
                    return Q

    return Q


if not os.path.exists('q_value_function_3x3.pickle'):
    with open('q_value_function_3x3.pickle', 'wb') as handle:
        pickle.dump(BoardSaver(3), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('q_value_function_3x3.pickle', 'rb') as handle:
    q_value_function = pickle.load(handle)

print(len(q_value_function.boards))

for e in range(10):
    print(e)
    with open('q_value_function_3x3.pickle', 'rb') as handle:
        training_q_value_function = pickle.load(handle)

    env = DotsAndBoxes(2, DotsAndBoxesMaxIfKnownPolicy(training_q_value_function))
    q_value_function = q_learning(env, 2000, alpha=0.05, gamma=0.95, eps=0.1, epsmin=0.01, eps_decay=.999995, Q=q_value_function)
    del training_q_value_function
    with open('q_value_function_3x3.pickle', 'wb') as handle:
        pickle.dump(q_value_function, handle, protocol=pickle.HIGHEST_PROTOCOL)

inp = ""
env = DotsAndBoxes(2, DotsAndBoxesMaxIfKnownPolicy(q_value_function))
env.render()
while inp != "quit":
    inp = sys.stdin.readline()

    def splitter(n):
        node = list(map(int, n.split(",")))
        return node[0], node[1]


    action = list(map(splitter, inp.split(" ")))
    observation, reward, done, _ = env.step(action)

    env.render()
    if done:
        env.reset()
        env.render()
