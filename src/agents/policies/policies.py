import random

class DotsAndBoxesPolicy:
    def __init__(self, q_value_function):
        self._q_value_function = q_value_function

    def next_action(self, state, action_space):
        raise NotImplementedError()


class DotsAndBoxesCloseBoxesPolicy(DotsAndBoxesPolicy):
    """
    Closes a box if possible, if not, returns a random choice of action space
    """

    def next_action(self, state, action_space):
        max_width = max(map(lambda x: x[1][1], action_space))
        max_height = max(map(lambda x: x[1][0], action_space))
        for i in range(max_height):
            for j in range(max_width):
                borders = [
                    ((i, j), (i, j + 1)),
                    ((i, j), (i + 1, j)),
                    ((i + 1, j), (i + 1, j + 1)),
                    ((i, j + 1), (i + 1, j + 1)),
                ]
                # If there is a box with 3 sides closed, close the last one
                if sum(1 for b in borders if b not in action_space) == 3:
                    return next(b for b in borders if b in action_space)

        return random.sample(sorted(action_space), 1)[0]


class DotsAndBoxesRandomPolicy(DotsAndBoxesPolicy):
    """
    Returns a random choice of action space
    """
    def next_action(self, state, action_space):
        return random.sample(sorted(action_space), 1)[0]


class DotsAndBoxesMaxIfKnownPolicy(DotsAndBoxesPolicy):
    """
    Return an action with maximum reward if state is known.
    Return a random action in other case.
    """
    def next_action(self, state, action_space):
        if self._q_value_function.contains(state):
            action = max(action_space, key=lambda a: self._q_value_function.get(state, a))
        else:
            action = random.sample(sorted(action_space), 1)[0]
        return action

class DotsAndBoxesMixerPolicy(DotsAndBoxesPolicy):
    """
    Return an action with maximum reward if state is known.
    Return a random action in other case.
    """
    def __init__(self, q_value_function):
        super().__init__(q_value_function)
        self._greedy = DotsAndBoxesCloseBoxesPolicy(q_value_function=q_value_function)

    def next_action(self, state, action_space):
        
        if self._q_value_function.contains(state):
            action = max(action_space, key=lambda a: self._q_value_function.get(state, a))
        else:
            action = self._greedy.next_action(state, action_space)
        return action

    def update_q_value_function(self, q_value_function):
        self._q_value_function = q_value_function