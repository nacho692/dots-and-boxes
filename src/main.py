import sys
import logging
import numpy as np
import random
import pickle
import os.path
from collections import defaultdict

from learning_player import BoardSaver
from dots_boxes import (
    DotsAndBoxes,
    DotsAndBoxesMaxIfKnownPolicy,
    DotsAndBoxesRandomPolicy,
    DotsAndBoxesState,
    DotsAndBoxesCloseBoxesPolicy,
    DotsAndBoxesMixerPolicy
)
import logging

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

[]
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
    won = 0
    new_states = defaultdict(int)
    amount_of_turns = defaultdict(int)
    for e in range(num_episodes):
        state = env.reset()
        turn = 0
        if not Q.contains(state):
            for a in env.action_spaces:
                Q.define(state, a, env.size)
            new_states[turn] += 1

        action = epsilon_greedy(Q, state, env.action_spaces, eps)
        done = False
        
        while not done:
            next_state, info = env.step(action)
            turn += 1
            reward = info.get("reward")
            done = info.get("done")
            rw += reward

            if done:
                next_state = None
                won += info.get("player_1_points") > info.get("player_2_points")

            elif not Q.contains(next_state):
                for a in env.action_spaces:
                    Q.define(next_state, a, env.size)
                new_states[turn] += 1
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
        amount_of_turns[turn] += 1
        if e % 100 == 0:
            avg_rw = rw / 100
            avg_won = won / 100
            average_new_states = sum((key*value for key, value in new_states.items())) / sum(new_states.values())
            average_amount_of_turns = sum((key*value for key, value in amount_of_turns.items())) / sum(amount_of_turns.values())
            logging.info(
                f"episode: {e}, reward rate: {avg_rw}, new states: {sum(new_states.values())}, epsilon: {eps}, avg_won: {avg_won}. avg_pd_ns {average_new_states}, avg_pd_aot {average_amount_of_turns}"
            )
            
            rw = 0
            won = 0
            if avg_won > 0.65:
                logging.info(f"Update q value function: episode: {e}, reward rate: {avg_rw}, new states: {sum(new_states.values())}, epsilon: {eps}, avg_won: {avg_won}. avg_pd_ns {average_new_states}, avg_pd_aot {average_amount_of_turns}")
                env.update_q_value_function(q_value_function=Q)
                q_file = f"q_value_function_{board_size}x{board_size}_epoch{e}.pickle"
                save_value_function(q_file, Q)
            new_states = defaultdict(int)
            amount_of_turns = defaultdict(int)
    
    return Q


def play_against_player(board_size, q_value_function):
    inp = ""
    env = DotsAndBoxes(board_size, DotsAndBoxesMixerPolicy(q_value_function))
    env.render()
    while inp != "quit":
        inp = sys.stdin.readline()

        def splitter(n):
            node = list(map(int, n.split(",")))
            return node[0], node[1]

        action = list(map(splitter, inp.split(" ")))
        observation, info = env.step(action)
        done = info.get("done")

        env.render()
        if done:
            env.reset()
            env.render()


def save_q(q_file, q_value_function: BoardSaver):
    with open(q_file, "wb") as handle:
        pickle.dump(q_value_function, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_q(q_file) -> BoardSaver:
    with open(q_file, "rb") as handle:
        training_q_value_function = pickle.load(handle)
    return training_q_value_function


def train(board_size, q_file, q_value_function):
    for e in range(100):
        logging.info(e)
        training_q_value_function = load_q(q_file)

        env = DotsAndBoxes(board_size, DotsAndBoxesMixerPolicy(training_q_value_function))
        q_value_function = q_learning(
            env, 2_000, alpha=0.05, gamma=0.95, eps=0.1, epsmin=0.01, eps_decay=0.999995, Q=q_value_function
        )
        del training_q_value_function
        save_q(q_file, q_value_function)
    return q_value_function


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    board_size = 3
    q_file = f"q_value_function_{board_size}x{board_size}.pickle"

    if not os.path.exists(q_file):
        with open(q_file, "wb") as handle:
            pickle.dump(BoardSaver(board_size), handle, protocol=pickle.HIGHEST_PROTOCOL)

    q_value_function = load_q(q_file)
    logging.info(len(q_value_function.boards))

    train(board_size, q_file, q_value_function)

    # play_against_player(board_size, q_value_function)
