import logging
import random
import numpy as np

from .policies import DotsAndBoxesPolicy
from src.dots_boxes import DotsAndBoxes, DotsAndBoxesState
from src.learning_player import BoardSaver

class MachineAgent:
    def __init__(self, env:DotsAndBoxes, policy:DotsAndBoxesPolicy):
        self._policy = policy
        self._env = env
    
    def get_action(self, observations: DotsAndBoxesState):
        return self._policy.next_action(observations, self._env.action_spaces), False
    
    def update(self, *args, **kwargs):
        pass

    def update_value_function(self, Q: BoardSaver):
        self._policy._q_value_function = Q.copy()

class UserAgent:
    def __init__(self, env: DotsAndBoxes):
        self._env = env
    
    def get_action(self, observations: DotsAndBoxesState):
        user_input = input(f'give two adjacent nodes in the format "0,0 0,1": ')
        user_input = user_input.split(' ')
        action = [(int(a.split(',')[0]), int(a.split(',')[1])) for a in user_input]
        return action, False
    
    def update(self, *args, **kwargs):
        pass

class EpsilonGreedyAgent:
    def __init__(self,
                 env: DotsAndBoxes,
                 policy: DotsAndBoxesPolicy,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 gamma: float,
                ):
        self._learning_rate = learning_rate
        self._policy = policy
        self._env = env
        self._initial_epsilon = initial_epsilon
        self._epsilon_decay = epsilon_decay
        self._final_epsilon = final_epsilon
        self.epsilon = initial_epsilon
        self._gamma = gamma
        self.training_error = []


    def get_action(self, observations: DotsAndBoxesState):
        action_spaces = self._env.action_spaces
        new_state = self.set_action_initial_value(observations, action_spaces)
        
        if np.random.random() < self.epsilon:
            action = random.choices(list(action_spaces))[0]
        else:
            action = max(action_spaces, key=lambda a: self._policy._q_value_function.get(observations, a))
        return action, new_state

    def set_action_initial_value(self, observations: DotsAndBoxesState, action_spaces):
        if not self._policy._q_value_function.contains(observations):
            for a in action_spaces:
                self._policy._q_value_function.define(observations, a, self._env.size)
            return True
        return False
    
    def update(
    self,
    state: DotsAndBoxesState,
    action,
    reward: float,
    terminated: bool,
    next_state: DotsAndBoxesState,
    ):
        """Updates the Q-value of an action."""
        old_q_value = self._policy._q_value_function.get(state, action)
        self.set_action_initial_value(next_state, self._env.action_spaces)
        next_expected_value = (
            max(map(lambda a: self._policy._q_value_function.get(next_state, a), self._env.action_spaces)) if next_state is not None and not terminated else 0
        )
        temporal_difference = reward + self._gamma * next_expected_value - old_q_value
        new_q_value = old_q_value + self._learning_rate * temporal_difference
        self._policy._q_value_function.define(state, action, new_q_value)
        self.training_error.append(temporal_difference)
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self._final_epsilon, self.epsilon - self._epsilon_decay)