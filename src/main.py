import sys
import logging
import numpy as np
import random
import pickle
import os.path
from collections import defaultdict

from .learning_player import BoardSaver
from .dots_boxes import (
    DotsAndBoxes,
    DotsAndBoxesState,
)
from .agents import MachineAgent, UserAgent, EpsilonGreedyAgent
from .agents.policies import DotsAndBoxesCloseBoxesPolicy, DotsAndBoxesMixerPolicy

from .constants import PLAYER_1, PLAYER_2

import logging

random.seed(0)


def q_learning(
    env: DotsAndBoxes,
    num_episodes: int,
    machine_agent: MachineAgent,
    learning_agent: EpsilonGreedyAgent,
) -> BoardSaver:
    rw = 0
    won = 0
    amount_of_turns = defaultdict(int)
    
    for e in range(num_episodes):
        reward_game, won_game, turns = play_one_game(env, machine_agent, learning_agent)
        rw += reward_game
        won += won_game
        amount_of_turns[turns] += 1
        
        if e % 100 == 0:
            avg_rw = rw / 100
            avg_won = won / 100
            #average_new_states = sum((key*value for key, value in new_states.items())) / sum(new_states.values())
            average_amount_of_turns = sum((key*value for key, value in amount_of_turns.items())) / sum(amount_of_turns.values())
            logging.info(
                f"episode: {e}, reward rate: {avg_rw:.2f}, avg_won: {avg_won:.2%}. avg_pd_aot {average_amount_of_turns:.2f}"
            )
            
            
            if avg_won > 0.65:
                logging.info(f"Update q value function: episode: {e}, reward rate: {avg_rw:.2f}, avg_won: {avg_won:.2%}. avg_pd_aot {average_amount_of_turns:.2%}")
                machine_agent.update_value_function(learning_agent._policy._q_value_function)
                q_file = f"q_value_function_{board_size}x{board_size}_epoch{e}.pickle"
                save_value_function(q_file, learning_agent._policy._q_value_function)
            rw = 0
            won = 0
            amount_of_turns = defaultdict(int)

    return learning_agent._policy._q_value_function

def play_one_game(env: DotsAndBoxes, machine_agent, learning_agent, render=False):
    state, info = env.reset()
    turn = 0
    rw = 0
    done = False
    
    while not done:
        if render:
            env.render()
        player_turn = env._player_turn
        if player_turn == PLAYER_1:
            player = learning_agent
        else:
            player = machine_agent
        action = player.get_action(state)
        next_state, reward, done, _, info = env.step(action)
        player.update(state, action, reward, done, next_state)
            
        turn += 1
        if player_turn == PLAYER_1:
            rw += reward
        state = next_state
    
    if render:
        env.render()
    won = info.get("player_1_points") > info.get("player_2_points")
    return rw, won, turn


def play_against_player(board_size, q_file):
    q_value_function = load_value_function(q_file)
    env = DotsAndBoxes(board_size)
    machine_agent = MachineAgent(env, DotsAndBoxesMixerPolicy(q_value_function))
    user_agent = UserAgent(env)
    while True:      
        play_one_game(env, machine_agent, user_agent, render=True)
            


def save_value_function(q_file, q_value_function: BoardSaver):
    with open(q_file, "wb") as handle:
        pickle.dump(q_value_function, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_value_function(q_file) -> BoardSaver:
    with open(q_file, "rb") as handle:
        training_q_value_function = pickle.load(handle)
    return training_q_value_function


def train(epochs: int, matches_per_epoch: int, board_size: int, q_file:str):
    q_value_function = load_value_function(q_file)
    env = DotsAndBoxes(board_size)
    greedy_policy = DotsAndBoxesCloseBoxesPolicy(q_value_function=q_value_function)
    machine_agent = MachineAgent(env=env, policy=greedy_policy)
    
    for e in range(epochs):
        if  q_value_function is None:
            q_value_function = BoardSaver(board_size)
        learning_policy = DotsAndBoxesMixerPolicy(q_value_function=q_value_function)
        learning_agent = EpsilonGreedyAgent(env=env, policy=learning_policy, learning_rate=0.05, initial_epsilon=0.1, epsilon_decay=0.999995, final_epsilon=0.01, gamma=0.95)
        
        q_value_function = q_learning(
            env, matches_per_epoch, machine_agent, learning_agent
        )
        
        save_value_function(q_file, q_value_function)
        break

    logging.info("finished training")

def policy_against_policy(board_size, games_to_play, agent_1_factory, agent_2_factory):
    env = DotsAndBoxes(board_size)
    agent_1 = agent_1_factory(env)
    agent_2 = agent_2_factory(env)
    reward = 0
    won = 0
    logging.info(f"start policy vs policy")
    for _ in range(games_to_play):
        game_rw, game_won, turn = play_one_game(env, agent_1, agent_2)        
        game_rw += game_rw
        won += int(game_won)
    logging.info(f"[{type(agent_2).__name__}] won {won/games_to_play:.2%} of the games with an average reward of {reward/games_to_play:.2%} against {[type(agent_1).__name__]}")
    

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    board_size = 3
    epochs = 100
    matches_per_epoch = 2_000
    q_file = f"q_value_function_{board_size}x{board_size}.pickle"

    if not os.path.exists(q_file):
        with open(q_file, "wb") as handle:
            pickle.dump(BoardSaver(board_size), handle, protocol=pickle.HIGHEST_PROTOCOL)

        
    train(epochs, matches_per_epoch, board_size, q_file)
    # epoch = 439600
    # q_file = f"q_value_function_{board_size}x{board_size}_epoch{epoch}.pickle"
    # play_against_player(board_size, q_file)
    # mixer_factory = lambda q_file: DotsAndBoxesMixerPolicy(load_value_function(q_file))
    # machine_agent_factory = lambda env, q_file: MachineAgent(env=env, policy=mixer_factory(q_file))
    # for epoch in [312900, 404600, 414900, 430000, 439600]:
    #     logging.info(f"best solution is {epoch}")
    #     agent_1_factory = lambda env: machine_agent_factory(env, f"q_value_function_{board_size}x{board_size}.pickle")
    #     agent_2_factory = lambda env: machine_agent_factory(env, f"q_value_function_{board_size}x{board_size}_epoch{epoch}.pickle")
    #     games_to_play = 1000
    #     policy_against_policy(board_size, games_to_play, agent_1_factory, agent_2_factory)
        