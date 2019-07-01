import sys
from learning_player import *
from dots_boxes import DotsAndBoxes
import numpy as np
import random
import pickle
import os.path
import time


def epsilon_greedy(Q_s, action_spaces, epsilon):
    max_action = max(Q_s.items(), key=lambda x: x[1])[0]
    probs = list(
        map(lambda x: 1 - epsilon + epsilon / len(action_spaces)\
            if max_action == x[0] else epsilon / len(action_spaces),
                     Q_s.items()))
    action = random.choices(list(action_spaces), probs).pop()
    return action


def update_q_sarsamax(alpha, gamma, Q, board, action, reward, next_board):
    if next_board is None:
        next_value = 0
    else:
        next_action = max(Q.get(next_board).items(), key=lambda item: item[1])
        next_value = next_action[1]
    return Q.get(board)[action] + alpha * (reward + gamma * next_value - Q.get(board)[action])


def q_learning(env, num_episodes, alpha, gamma=1.0, eps=1.0, eps_decay=.9999, epsmin=0.01, Q=None):
    if Q is None:
        Q = BoardHashSaver(env.size)
    rw = 0
    new_boards = 0
    over_1k = 5
    for e in range(num_episodes):
        board = env.reset()
        done = False
        if not Q.contains(board):
            Q.add(board, {action: np.random.rand() for action in env.action_spaces})
            new_boards += 1
        while not done:
            action = epsilon_greedy(Q.get(board), env.action_spaces, eps)
            next_board, reward, done, info = env.step(action)

            if done:
                next_board = None
            elif not Q.contains(next_board):
                Q.add(next_board, {action: np.random.rand() for action in env.action_spaces})
                new_boards += 1

            eps = max(epsmin, eps * eps_decay)
            Q.get(board)[action] = update_q_sarsamax(alpha, gamma, Q, board, action, reward, next_board)

            board = next_board
            rw += reward
        if e % 1000 == 0:
            avg_rw = rw/1000
            print("episode: {}, reward rate: {}, new boards: {}, epsilon: {}, over_1k: {}"
                  .format(e, avg_rw, new_boards, eps, over_1k))
            new_boards = 0
            rw = 0
            if avg_rw > 1:
                over_1k -= 1
                if over_1k == 0:
                    return Q

    return Q


if not os.path.exists('policy_3x3.pickle'):
    with open('policy_3x3.pickle', 'wb') as handle:
        pickle.dump(BoardHashSaver(3), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('policy_3x3.pickle', 'rb') as handle:
    policy = pickle.load(handle)

for e in range(100):
    print(e)
    with open('policy_3x3.pickle', 'rb') as handle:
        training_policy = pickle.load(handle)

    env = DotsAndBoxes(3, training_policy)
    policy = q_learning(env, 9999999, alpha=0.05, gamma=0.99, eps=0.2, epsmin=0.05, eps_decay=.9999995, Q=policy)
    del training_policy
    with open('policy_3x3.pickle', 'wb') as handle:
        pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

inp = ""
env = DotsAndBoxes(3, policy)
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
