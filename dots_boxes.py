import pygame
import gym
import itertools
from gym import spaces, logger
from gym.utils import seeding


class DotsAndBoxes(gym.Env):
    class Box():

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

    class Node():

        def __init__(self, position):
            self.index = position[0] + position[1] * 4
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

    def __init__(self, size=3):

        self.n = (size + 1) * (size + 1)
        self.size = size
        self.edges = [[DotsAndBoxes.Node((i, j)) for j in range(self.size + 1)] for i in range(self.size + 1)]
        self.boxes = [[None for j in range(self.size)] for i in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                corners = set()
                corners.add(self.edges[i][j])
                corners.add(self.edges[i][j + 1])
                corners.add(self.edges[i + 1][j])
                corners.add(self.edges[i + 1][j + 1])
                self.boxes[i][j] = DotsAndBoxes.Box((i, j), corners)

        self.points_player_1 = 0
        self.points_player_2 = 0

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
        """
        pos_i = action[0]
        pos_j = action[1]

        assert abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1]) == 1, "Nodes are not adjacent"

        node_i = self.edges[pos_i[0]][pos_i[1]]
        node_j = self.edges[pos_j[0]][pos_j[1]]

        assert not node_i.is_connected(node_j), "The edge already exists"

        node_i.connect_to(node_j, 1)

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

        for node in itertools.chain.from_iterable(self.edges):
            n_s_pos = self.get_node_screen_position(node.position)
            for other_node in node.connected_nodes:
                if other_node.index < node.index:
                    pass
                on_s_pos = self.get_node_screen_position(other_node.position)
                pygame.draw.line(self.screen, GREEN, n_s_pos, on_s_pos, width=1)

            pygame.draw.circle(self.screen, RED, n_s_pos, 5, width=5)
            text = self.font.render('({},{})'.format(node.position[0], node.position[1]), True, WHITE, BLACK)
            self.screen.blit(text, (n_s_pos[0] - 16, n_s_pos[1] + 10))

        for box in itertools.chain.from_iterable(self.boxes):
            text = self.font.render(str(box.controller), True, WHITE, BLACK)
            self.screen.blit(text, self.get_box_screen_position(box.position))

        pygame.display.update()

    def get_box_screen_position(self, pos):
        corner_position = self.get_node_screen_position(pos)
        box_position = (corner_position[0] + self.box_step // 2, corner_position[1] + self.box_step // 2)

        return box_position

    def get_node_screen_position(self, pos):
        i, j = pos

        starting_point = (self.screen_width // 2 - self.display_size // 2,
                          self.screen_height // 2 - self.display_size // 2)
        dot_position = (starting_point[0] + i * self.box_step, starting_point[1] + j * self.box_step)

        return dot_position
