import gym
import pygame
import random
import itertools
from typing import NamedTuple
from .constants import PLAYER_1, PLAYER_2

class DotsAndBoxesState(NamedTuple):
    state: list
    player_points: int

    def __hash__(self):
        return hash(tuple(self.state) + tuple(self.player_points))

    def __eq__(self, other):
        return self.state == other.state and self.player_points == other.player_points

    def __str__(self):
        return "State: {}, Action: {}".format(self.state, self.player_points)


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

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, size=3):

        self.n = (size + 1) * (size + 1)
        self.size = size
        self.nodes = []
        self.boxes = []
        self.done = False
        self.action_spaces = set()
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
        """Executes a selected action

        Args:
            action: A tuple of positions. A position is a tuple indicating the node by position, (x, y) indicating
            the nodes to connect
        Returns:
            observation (object): Agent's observation of the current environment.
            reward (float) : amount of reward returned after previous action
            terminated (bool): whether the episode has ended, If this happens the user needs  to call reset() to use a new env
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied. 
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        player_1_old_points = self._player_points(PLAYER_1)
        player_2_old_points = self._player_points(PLAYER_2)

        player_turn = self.execute_action(action)

        player_1_points = self._player_points(PLAYER_1)
        player_2_points = self._player_points(PLAYER_2)

        new_player_1_points = player_1_points - player_1_old_points
        new_player_2_points = player_2_points - player_2_old_points

        total_boxes = self.size * self.size
        self.done = (
            max(player_1_points, player_2_points) > total_boxes // 2 or player_1_points + player_2_points == total_boxes
        )

        reward = new_player_1_points - new_player_2_points

        if self.done:
            if player_1_points > player_2_points:
                reward += total_boxes
            elif player_1_points < player_2_points:
                reward -= total_boxes

        info = {
            "player_1_points": player_1_points,
            "player_2_points": player_2_points,
            "new_player_1_points": new_player_1_points,
            "new_player_2_points": new_player_2_points,
            "reward": reward,
            "done": self.done,
            "player_turn": player_turn
        }

        return self._get_current_observation(), reward, self.done, self.done, info

    def execute_action(self, action):
        has_new_box = self._player_pick(self._player_turn, action)
        if not has_new_box:
            self.next_player()
        return self._player_turn
        
    def next_player(self):
        self._player_turn = 2 - ((self._player_turn + 1) % 2)
    

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
        def get_edges(u):
            return ((u.position, v.position) for v in u.connected_nodes if v.index > u.index)

        return DotsAndBoxesState(
            state=list(itertools.chain.from_iterable(map(get_edges, itertools.chain.from_iterable(self.nodes)))),
            player_points=self._player_points(self._player_turn),
        )

    def render(self, mode="human"):

        # PyGame screen
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)

        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 0, 32)
            self.font = pygame.font.Font("freesansbold.ttf", 16)
            pygame.display.set_caption("Markov Dots and Boxes")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        self.screen.fill(BLACK)


        for node in itertools.chain.from_iterable(self.nodes):
            n_s_pos = self._get_node_screen_position(node.position)
            for other_node in node.connected_nodes:
                if other_node.index < node.index:
                    pass
                on_s_pos = self._get_node_screen_position(other_node.position)
                pygame.draw.line(self.screen, GREEN, n_s_pos, on_s_pos, width=1)

            pygame.draw.circle(self.screen, RED, n_s_pos, 5, width=5)
            text = self.font.render("({},{})".format(node.position[0], node.position[1]), True, WHITE, BLACK)
            self.screen.blit(text, (n_s_pos[0] - 16, n_s_pos[1] + 10))

        for box in itertools.chain.from_iterable(self.boxes):
            text = self.font.render(str(box.get_controller()), True, WHITE, BLACK)
            self.screen.blit(text, self._get_box_screen_position(box.position))
        pygame.display.update()

    def reset(self) -> tuple[DotsAndBoxesState, dict]:
        self.nodes = [
            [DotsAndBoxes.Node((i, j), i + j * self.size) for j in range(self.size + 1)] for i in range(self.size + 1)
        ]
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
        
        player_start = PLAYER_1
        if random.choice([True, False]):
            player_start = PLAYER_2
        info = {
            'player_start': player_start
        }
        self._player_turn = player_start
        return self._get_current_observation(), info

    def _get_box_screen_position(self, pos):
        corner_position = self._get_node_screen_position(pos)
        box_position = (corner_position[0] + self.box_step // 2, corner_position[1] + self.box_step // 2)

        return box_position

    def _get_node_screen_position(self, pos):
        i, j = pos

        starting_point = (
            self.screen_width // 2 - self.display_size // 2,
            self.screen_height // 2 - self.display_size // 2,
        )
        dot_position = (starting_point[0] + i * self.box_step, starting_point[1] + j * self.box_step)

        return dot_position
