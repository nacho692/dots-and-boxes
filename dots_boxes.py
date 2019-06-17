import pygame
import gym
from gym import spaces, logger
from gym.utils import seeding


class DotsAndBoxes(gym.Env):

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}
	def __init__(self, size = 3):
		self.n = (size+1)*(size+1)
		self.size = size
		self.screen = None
		self.edges = [[] for i in range(self.n)]
		self.box_rewarded = [[0 for _ in range(self.size)] for i in range(self.size)]

	def step(self, action):
		""" Executes a selected action

		Args:
			action: A tuple of positions. A position is a tuple indicating the node by position, (x, y) indicating
			the nodes to connect
		"""

		node_i = action[0]
		node_j = action[1]
		index_i = node_i[1]*(self.size + 1) + node_i[0]
		index_j = node_j[1]*(self.size + 1) + node_j[0]
		self.edges[index_i].append(index_j)
		self.edges[index_j].append(index_i)
		print(self.edges)

	def render(self, mode='human'):
        
    	# PyGame screen
		BLACK = (0, 0, 0)
		WHITE = (255, 255, 255)
		GREEN = (0, 255, 0)
		RED = (255, 0 , 0)

		screen_width = 640
		screen_height = 480

		if self.screen is None:
			pygame.init()
			pygame.font.init()
			self.screen = pygame.display.set_mode((screen_width, screen_height), 0, 32)
			self.font = pygame.font.Font('freesansbold.ttf', 16)
			pygame.display.set_caption("Markov Dots and Boxes")

		self.screen.fill(BLACK)
		

		for i in range(self.n):
			i_pos = self.get_node_position(i, screen_height, screen_width)
			pygame.draw.circle(self.screen, RED, i_pos, 5, width=5)
			text = self.font.render('({},{})'.format(i % (self.size + 1), i // (self.size + 1)), True, WHITE, BLACK)
			self.screen.blit(text, (i_pos[0] - 16, i_pos[1] + 10))

			for j in [ j for j in self.edges[i] if j > i]:
				j_pos = self.get_node_position(j, screen_height, screen_width)
				pygame.draw.line(self.screen, GREEN, i_pos, j_pos, width=1)

		pygame.display.update()

	def get_node_position(self, n, screen_height, screen_width):
		i, j = (n % (self.size + 1), n // (self.size + 1))
		margin_size = 40		
		box_size = min(screen_height - margin_size*2, screen_width - margin_size*2)
		step = box_size // self.size

		starting_point = (screen_width//2 - box_size//2, screen_height//2 - box_size//2)
		dot_position = (starting_point[0] + i*step, starting_point[1] + j*step)

		return dot_position

