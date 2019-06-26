class Board():
	"""
	Represent the board as an array of booleans where each dot has 2 bools.
	Dots indexes in array are in (0,0), (0,1), ..., (<last_row>, <last_col>) order.
	A pair of booleans for a dot means <edge_going_down>, <edge_going_to_the_right> respectively.
	The value of the boolean indicates whether the edge has been added by a player or not. 
	"""
	def __init__(self, count_rows, count_cols):
		self.count_rows = count_rows
		self.count_cols = count_cols
		self.taken_edges = self.clean_board()
		return

	def clean_board(self):
		"""
		Return a new default board game representation with no taken edge.
		"""
		return [False] * self.count_cols * self.count_rows * 2

	def check_coordinate_bound(self, coordinate):
		"""
		Check coordinate is inside rows/cols board bounds.
		"""
		if self.count_rows <= coordinate[0] or self.count_cols <= coordinate[1]:
			raise Exception("Invalid coordinate {} for rows {} and cols {}".format(coordinate, self.count_rows, self.count_cols))
		return

	def check_existing_edge(self, first_coordinate, second_coordinate):
		"""
		Check there is an edge between coordinates
		"""
		if first_coordinate == second_coordinate:
			raise Exception("first_coordinate {} equals second_coordinate {}".format(first_coordinate, second_coordinate))

		if second_coordinate[0] < first_coordinate[0] or second_coordinate[1] < first_coordinate[1]:
			raise Exception("first_coordinate {} is not on top or left of second_coordinate {}".format(first_coordinate, second_coordinate))

		if first_coordinate[0] not in range(second_coordinate[0] - 1, second_coordinate[0] + 1) or first_coordinate[1] not in range(second_coordinate[1] - 1, second_coordinate[1] + 1):
			raise Exception("first_coordinate {} is contiguous to second_coordinate {}".format(first_coordinate, second_coordinate))
		return


	def get_board_position(self, first_coordinate, second_coordinate):
		_second_coordinate_is_on_right = first_coordinate[1] == second_coordinate[1] + 1

		position = first_coordinate[0] * self.count_cols + 2 * first_coordinate[1] # each node has two indexes in the board
		position += 0 if _second_coordinate_is_on_right else 1 # bits are ordered as right and down edges respectively

		# only to check errors during development
		for _coordinate in [first_coordinate, second_coordinate]:
			self.check_coordinate_bound(_coordinate)

		self.check_existing_edge(first_coordinate, second_coordinate)

		return position

	def update_taken_edges(self, taken_edges):
		"""
		taken_edges :list: of edges, where and edge is a pair of coordinates and a coordinate a pair of int
		"""
		for (_first_coordinate, _second_coordinate) in taken_edges:
			self.taken_edges[self.get_board_position(_first_coordinate, _second_coordinate)] = True
		return

	def get_ordered_edges(self):
		"""
		Yields a boolean for each edge indicating wether it is or not used.
		Edges are traversed in right-down order in (0,0), (0,1), ... (last_row, last_col) order.
		"""
		for _edge in self.taken_edges:
			yield _edge
		return

class BoardSaver():
	"""
	A bit trie set that stores all the boards so as to compare for existence quickly.
	"""
	def __init__(self):
		self.bit_trie = {}
		return

	def contains_board(self, board):
		"""
		Parameters:
		board (Board)

		Returns:
		bool indicating whether is included or not. 
		"""
		is_contained = True
		root = self.bit_trie

		for _taken_edge in board.get_ordered_edges():
			if _taken_edge in root.keys() and is_contained:
				root = root[_taken_edge] # keep traversing the trie
			else:
				is_contained = False
				break

		return is_contained

	def add_board(self, board):
		"""
		Add a board to the set.

		Parameters:
		board (Board)
		"""
		root = self.bit_trie

		for _taken_edge in board.get_ordered_edges():
			if _taken_edge not in root.keys():
				root[_taken_edge] = {}
			
			root = root[_taken_edge]

		return

class GameStatus():
	"""
	Represent a status of a game: board, taken edges, players' points.

	Board is represented as a vector of bools where each coordenate in the map has two bits,
	one for its right edge (if any) and a second bit for its down edge (if any).
	"""
	def __init__(self, count_rows, count_cols):
		self.count_rows = count_rows
		self.count_cols = count_cols
		self.points_player_1 = 0
		self.points_player_2 = 0
		self.board = Board(self.count_rows, self.count_cols)
		return

	def save_players_points(self, points_player_1, points_player_2):
		self.points_player_1 = points_player_1
		self.points_player_2 = points_player_2
		return

	def stores_entire_state(self, points_player_1, points_player_2, taken_edges):
		"""
		points_player_1 :int:
		points_player_2 :int:
		taken_edges :list: of edges, where and edge is a pair of coordinates and a coordinate a pair of int
		"""
		self.save_players_points(points_player_1, points_player_2)
		self.board = Board(self.count_rows, self.count_cols) # Initialize all edges as unused, represented with False value
		self.board.update_taken_edges(taken_edges)
		return

	def update_state_with_taken_edges(self, points_player_1, points_player_2, taken_edges):
		"""
		points_player_1 :int:
		points_player_2 :int:
		taken_edges :list: of edges, where and edge is a pair of coordinates and a coordinate a pair of int
		"""
		self.save_players_points(points_player_1, points_player_2)
		self.board.update_taken_edges(taken_edges)
		return
