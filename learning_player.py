import itertools

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

	def __hash__(self):
		mask = 0
		for _ith, _bool in enumerate(self.taken_edges):
			mask &= _bool << _ith
		return mask

	def __eq__(self):
		_are_equal = self.count_rows == other_game_status.count_rows and self.count_cols == other_game_status.count_cols
		
		for _bit_izq, _bit_der in zip(self.board.get_ordered_taken_edges(), other_game_status.board.get_ordered_taken_edges()):
			if not _are_equal and _bit_izq != _bit_der:
				_are_equal = False
				break

		return _are_equal

	def clean_board(self):
		"""
		Return a new default board game representation with no taken edge.
		"""
		return [False] * self.count_cols * self.count_rows * 2

	def __check_coordinate_bound(self, coordinate):
		"""
		Check coordinate is inside rows/cols board bounds.
		"""
		if self.count_rows <= coordinate[0] or self.count_cols <= coordinate[1]:
			raise Exception("Invalid coordinate {} for rows {} and cols {}".format(coordinate, self.count_rows, self.count_cols))
		return

	def __check_existing_edge(self, first_coordinate, second_coordinate):
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
			self.__check_coordinate_bound(_coordinate)

		self.__check_existing_edge(first_coordinate, second_coordinate)

		return position

	def update_taken_edges(self, taken_edges):
		"""
		taken_edges :list: of edges, where and edge is a pair of coordinates and a coordinate a pair of int
		"""
		for (_first_coordinate, _second_coordinate) in taken_edges:
			self.taken_edges[self.get_board_position(_first_coordinate, _second_coordinate)] = True
		return

	def get_ordered_taken_edges(self):
		"""
		Yields a boolean for each edge indicating wether it is or not used.
		Edges are traversed in right-down order in (0,0), (0,1), ... (last_row, last_col) order.
		"""
		for _edge in self.taken_edges:
			yield _edge
		return

	def __get_all_edges(self):
		for _coordinate in itertools.product(range(self.count_rows-1), range(self.count_cols-1)): # inner edges
			yield (_coordinate, (_coordinate[0] + 1, _coordinate[1]))
			yield (_coordinate, (_coordinate[0], _coordinate[1] + 1))

		for _row in range(self.count_rows - 1): # last row
			yield ((_row, self.count_cols - 1), (_row + 1, self.count_cols - 1))

		for _col in range(self.count_rows-1): # last column
			yield ((self.count_row - 1, _col), (self.count_cols - 1, _col + 1))

	def __rotate_coordinate(self, coordinate):
		print(coordinate)
		print((coordinate[1], (self.count_cols - 1 - coordinate[0])))
		return (coordinate[1], (self.count_cols - 1 - coordinate[0]))

	def __rotate_edge(self, lft_coordinate, rgt_coordinate):
		_next_lft_coordinate = self.__rotate_coordinate(lft_coordinate)
		_next_rgt_coordinate = self.__rotate_coordinate(rgt_coordinate)

		if _next_rgt_coordinate[0] < _next_lft_coordinate[0] or _next_rgt_coordinate[1] < _next_lft_coordinate[1]: # swap
			_next_lft_coordinate = _next_rgt_coordinate
			_next_rgt_coordinate = _next_lft_coordinate

		return (_next_lft_coordinate, _next_rgt_coordinate)

	def rotations(self):
		yield self

		for _ in range(3): # just rate
			_rotate_board = self.clean_board()
			
			for _actual_edge in self.__get_all_edges():
				_rotated_edge = self.__rotate_edge(*_actual_edge)
				print("original {} rotated {}".format(_actual_edge, _rotated_edge))
				_rotate_board[self.get_board_position(*_rotated_edge)] = self.taken_edges[self.get_board_position(*_actual_edge)]

			yield _rotate_board
		
		return

class BoardSaver():
	def contains_board(self, board):
		raise Exception()
	
	def add_board(self, board):
		raise Exception()
	
	def equivalent_board(self, board):
		for rotated_board in board.rotations():
			if self.contains_board(rotated_board):
				return rotated_board
		
		return board

class BoardHashSaver(BoardSaver):
	"""
	A bit trie set that stores all the boards so as to compare for existence quickly.
	"""
	def __init__(self):
		self.boards = {}
		return

	def contains_board(self, board):
		"""
		Parameters:
		board (Board)

		Returns:
		bool indicating whether is included or not. 
		"""
		return board in self.boards

	def add_board(self, board):
		"""
		Add a board to the set.

		Parameters:
		board (Board)
		"""
		equivalent_board = board # self.equivalent_board(board)

		if not self.contains_board(equivalent_board):  # only add it if it is new
			self.boards[equivalent_board] = 0
		
		return

class BoardTrieSaver(BoardSaver):
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

		for _taken_edge in board.get_ordered_taken_edges():
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
		equivalent_board = board # self.equivalent_board(board)

		if not self.contains_board(equivalent_board): # only add it if it is new
			root = self.bit_trie

			for _taken_edge in board.get_ordered_taken_edges():
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
		self.board = Board(self.count_rows, self.count_cols)
		return

	def __hash__(self):
		_hash = hash(self.board)
		_hash += self.count_rows * 23
		_hash += self.count_cols * 27
		return _hash

	def __eq__(self, other_game_status):
		return self.count_rows == other_game_status.count_rows \
		and self.count_cols == other_game_status.count_cols \
		and self.board == other_game_status.board

	def stores_entire_state(self, taken_edges):
		"""
		taken_edges :list: of edges, where and edge is a pair of coordinates and a coordinate a pair of int
		"""
		self.board = Board(self.count_rows, self.count_cols) # Initialize all edges as unused, represented with False value
		self.board.update_taken_edges(taken_edges)
		return

	def update_state_with_taken_edges(self, taken_edges):
		"""
		taken_edges :list: of edges, where and edge is a pair of coordinates and a coordinate a pair of int
		"""
		self.board.update_taken_edges(taken_edges)
		return

# example
# s = BoardHashSaver()
# gs = GameStatus(rows, cols)
# gs.update_state_with_taken_edges(lista_de_tuplas)
# gs.board tiene el tablero
# s = s.add_board(gs.board)