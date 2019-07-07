import pygame
import gym
import itertools
import random


class DotsAndBoxesPolicy:
    def __init__(self, q_value_function):
        self._q_value_function = q_value_function
        return

    def next_action(self, state, action_space):
        raise NotImplementedError()


class DotsAndBoxesRandomPolicy(DotsAndBoxesPolicy):
    """
    Returns a random choice of action space
    """
    def next_action(self, state, action_space):
        return random.sample(action_space, 1)[0]


class DotsAndBoxesMaxIfKnownPolicy(DotsAndBoxesPolicy):
    """
    Return an action with maximum reward if state is known.
    Return a random action in other case.
    """
    def next_action(self, state, action_space):
        if self._q_value_function.contains(state):
            action = max(action_space, key=lambda a: self._q_value_function.get(state, a))
        else:
            action = random.sample(action_space, 1)[0]
        return action


class DotsAndBoxes(gym.Env):
    class Box:
        """
        A Box is the object you want to capture more of to win the game. It has four corners (Nodes) and four sides
        which connect the Nodes.
        Whenever the four sides of the box are filled its "control" is handed to the last player to introduce a new
        side.
        """

        def __init__(self, position, corners):
            assert len(corners) == 4, "Each box has 4 corners"
            self.position = position
            self.corners = corners
            self.sides = set()
            self.controller = 0
            for c in corners:
                c.corner_to(self)

        def new_side(self, u, v, player):
            if u.index > v.index:
                self.sides.add((u, v))
            else:
                self.sides.add((v, u))

            if len(self.sides) == 4:
                self.controller = player

        def get_controller(self):
            return self.controller

    class Node:
        """
        A Node is one of the dots connected by edges. They are the corner of the Boxes.
        """

        def __init__(self, position, index):
            self.index = index
            self.position = position
            self.connected_nodes = set()
            self.boxes = set()

        def corner_to(self, box):
            self.boxes.add(box)
            assert len(self.boxes) <= 4, "Cannot have more than 4 boxes connected to a corner"

        def connect_to(self, node, player):
            self.connected_nodes.add(node)
            for b in self.boxes.intersection(node.boxes):
                b.new_side(self, node, player)

            if not node.is_connected(self):
                node.connect_to(self, player)

        def is_connected(self, node):
            return node in self.connected_nodes

        def __str__(self):
            return str(self.position)

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, size=3, policy=None):

        self.n = (size + 1) * (size + 1)
        self.size = size
        self.nodes = []
        self.boxes = []
        self.done = False
        self.action_spaces = set()
        self.policy = policy
        self.reset()

        # Rendering variables
        self.screen_width = 640
        self.screen_height = 480
        self.screen = None
        self.font = None
        self.margin_size = 40
        self.display_size = min(self.screen_height - self.margin_size * 2, self.screen_width - self.margin_size * 2)
        self.box_step = self.display_size // self.size

    def step(self, action):
        """ Executes a selected action

        Args:
            action: A tuple of positions. A position is a tuple indicating the node by position, (x, y) indicating
            the nodes to connect
        Returns:
            observation (object): Agent's observation of the current environment.
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        player_1_old_points = self._player_points(1)
        player_2_old_points = self._player_points(2)

        if not self._player_pick(1, action):
            self._player2()

        player_1_points = self._player_points(1)
        player_2_points = self._player_points(2)
        total_boxes = self.size * self.size
        self.done = max(player_1_points, player_2_points) > total_boxes//2 \
                    or player_1_points + player_2_points == total_boxes

        reward = (player_1_points - player_1_old_points) - (player_2_points - player_2_old_points)
        if self.done:
            if player_1_points > player_2_points:
                reward += total_boxes
            elif player_1_points < player_2_points:
                reward -= total_boxes

        return self._get_current_observation(), reward, self.done, None

    def _player2(self):
        new_point = True
        while new_point and len(self.action_spaces) > 0:
            state = self._get_current_observation()
            action = self.policy.next_action(state, self.action_spaces)
            new_point = self._player_pick(2, action)

    def _player_pick(self, player, action):
        old_player_points = self._player_points(player)
        assert not self.done
        pos_i = action[0]
        pos_j = action[1]

        assert abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1]) == 1, "Nodes are not adjacent"

        node_i = self.nodes[pos_i[0]][pos_i[1]]
        node_j = self.nodes[pos_j[0]][pos_j[1]]

        assert not node_i.is_connected(node_j), "The edge already exists"

        node_i.connect_to(node_j, player)
        if node_j.index > node_i.index:
            self.action_spaces.remove((node_i.position, node_j.position))
        else:
            self.action_spaces.remove((node_j.position, node_i.position))

        new_player_points = self._player_points(player)
        return old_player_points < new_player_points

    def _player_points(self, player):
        return len([1 for b in itertools.chain.from_iterable(self.boxes) if b.get_controller() == player])

    def _get_current_observation(self):
        get_edges = lambda u: ((u.position, v.position) for v in u.connected_nodes if v.index > u.index)

        return list(itertools.chain.from_iterable(map(get_edges, itertools.chain.from_iterable(self.nodes))))

    def render(self, mode='human'):

        # PyGame screen
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)

        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 0, 32)
            self.font = pygame.font.Font('freesansbold.ttf', 16)
            pygame.display.set_caption("Markov Dots and Boxes")

        self.screen.fill(BLACK)

        for node in itertools.chain.from_iterable(self.nodes):
            n_s_pos = self._get_node_screen_position(node.position)
            for other_node in node.connected_nodes:
                if other_node.index < node.index:
                    pass
                on_s_pos = self._get_node_screen_position(other_node.position)
                pygame.draw.line(self.screen, GREEN, n_s_pos, on_s_pos, width=1)

            pygame.draw.circle(self.screen, RED, n_s_pos, 5, width=5)
            text = self.font.render('({},{})'.format(node.position[0], node.position[1]), True, WHITE, BLACK)
            self.screen.blit(text, (n_s_pos[0] - 16, n_s_pos[1] + 10))

        for box in itertools.chain.from_iterable(self.boxes):
            text = self.font.render(str(box.get_controller()), True, WHITE, BLACK)
            self.screen.blit(text, self._get_box_screen_position(box.position))

        pygame.display.update()

    def reset(self):
        self.nodes = [[DotsAndBoxes.Node((i, j), i + j * self.size) for j in range(self.size + 1)]
                      for i in range(self.size + 1)]
        self.boxes = [[None for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                corners = set()
                corners.add(self.nodes[i][j])
                corners.add(self.nodes[i][j + 1])
                corners.add(self.nodes[i + 1][j])
                corners.add(self.nodes[i + 1][j + 1])
                self.boxes[i][j] = DotsAndBoxes.Box((i, j), corners)

        self.done = False
        self.action_spaces = set()
        for u in itertools.chain.from_iterable(self.nodes):
            if u.position[1] < self.size:
                self.action_spaces.add((u.position, (u.position[0], u.position[1] + 1)))
            if u.position[0] < self.size:
                self.action_spaces.add((u.position, (u.position[0] + 1, u.position[1])))

        if random.choice([True, False]):
            self._player2()

        return self._get_current_observation()

    def _get_box_screen_position(self, pos):
        corner_position = self._get_node_screen_position(pos)
        box_position = (corner_position[0] + self.box_step // 2, corner_position[1] + self.box_step // 2)

        return box_position

    def _get_node_screen_position(self, pos):
        i, j = pos

        starting_point = (self.screen_width // 2 - self.display_size // 2,
                          self.screen_height // 2 - self.display_size // 2)
        dot_position = (starting_point[0] + i * self.box_step, starting_point[1] + j * self.box_step)

        return dot_position
