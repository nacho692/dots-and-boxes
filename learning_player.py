import itertools


class Board:
	"""
	Represent the board as an array of booleans where each one indicate the existence of an edge.
	Edges are uniquly represented as (a,b)-(c,d), where a <= c and b <= d.
	Two valid edges are ordered as:
		Index (a,b)-(c,d) < Index (a2,b2)-(c2,d2) iff one of next are valid:
		- a < a2 
		- a == a2 amd b < b2
		- (a,b) == (a2,b2) and c < c2
	
	I.E., in a board of 2x2, edges are ordered:
	[(0,0)-(0,1), (0,0)-(1,0), (0,1)-(0,2), (0,1)-(1,1), (1,0)-(1,1), (1,0)-(2,1), (1,1)-(1,2), (1,1)-(2,1)]
	and their indexes are respectively 0, 1, ... , 2 * 2 * 2 - 1.
	
	Note that some of this edges won't be used, for example no edge which start in dot (1,1) can be selected. 
	"""
	def __init__(self, count_rows, count_cols, taken_edges=()):
		self.count_rows = count_rows
		self.count_cols = count_cols
		self.edges = self.clean_board()
		self.update_taken_edges(taken_edges)
		return

	def __hash__(self):
		mask = 0
		for _ith, _bool in enumerate(self.edges):
			mask |= _bool << _ith
		return mask

	def __eq__(self, other_board):
		_are_equal = self.count_rows == other_board.count_rows and self.count_cols == other_board.count_cols
		
		if other_board:

			for _bit_izq, _bit_der in zip(self.get_ordered_edges_values(), other_board.get_ordered_edges_values()):
				_are_equal = _bit_izq == _bit_der
				if not _are_equal:
					break
		return _are_equal

	def clean_board(self):
		"""
		Return a new default board game representation with no taken edge.
		"""
		return [False for _ in range(self.count_cols * self.count_rows * 2)]

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
		_share_row = first_coordinate[1] + 1 == second_coordinate[1]

		position = 2 * (first_coordinate[0] * self.count_cols + first_coordinate[1]) # each coordinate has two bools in the board
		position += 0 if _share_row else 1 # its bool for same row comes before the bool for same column

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
			self.edges[self.get_board_position(_first_coordinate, _second_coordinate)] = True
		return

	def get_ordered_edges_values(self):
		"""
		Yields a boolean for each edge indicating wether it is or not used.
		Edges are traversed in right-down order in (0,0), (0,1), ... (last_row, last_col) order.
		"""
		for _edge in self.edges:
			yield _edge
		return

	def __get_edges_coordinates(self):
		for _coordinate in itertools.product(range(self.count_rows-1), range(self.count_cols-1)): # inner edges
			yield (_coordinate, (_coordinate[0] + 1, _coordinate[1]))
			yield (_coordinate, (_coordinate[0], _coordinate[1] + 1))

		for _row in range(self.count_rows - 1): # last row
			yield ((_row, self.count_cols - 1), (_row + 1, self.count_cols - 1))

		for _col in range(self.count_rows-1): # last column
			yield ((self.count_rows - 1, _col), (self.count_cols - 1, _col + 1))

	def __rotate_coordinate(self, coordinate):
		return (-coordinate[1] + self.count_rows - 1, coordinate[0])

	def __rotate_edge(self, lft_coordinate, rgt_coordinate):
		_next_lft_coordinate = self.__rotate_coordinate(lft_coordinate)
		_next_rgt_coordinate = self.__rotate_coordinate(rgt_coordinate)

		if _next_rgt_coordinate[0] < _next_lft_coordinate[0] or _next_rgt_coordinate[1] < _next_lft_coordinate[1]: # swap
			_next_lft_coordinate, _next_rgt_coordinate = _next_rgt_coordinate, _next_lft_coordinate

		return (_next_lft_coordinate, _next_rgt_coordinate)

	def rotations(self):
		yield self
		yield self.reflect()

		_prev_rotated_board  = self

		for _ in range(3): # just rate
			_rotate_board = Board(self.count_rows, self.count_cols)
			
			for _actual_edge in self.__get_edges_coordinates():
				_rotated_edge = self.__rotate_edge(*_actual_edge)
				_rotate_board.edges[self.get_board_position(*_rotated_edge)] = _prev_rotated_board.edges[self.get_board_position(*_actual_edge)]

			yield _rotate_board
			yield _rotate_board.reflect()

			_prev_rotated_board = _rotate_board

		return

	def __reflect_coordinate(self, coordinate):
		return self.count_rows - 1 - coordinate[0], coordinate[1]

	def __reflect_edge(self, lft_coordinate, rgt_coordinate):
		_reflected_lft_coordinate = self.__reflect_coordinate(lft_coordinate)
		_reflected_rgt_coordinate = self.__reflect_coordinate(rgt_coordinate)

		if _reflected_rgt_coordinate[0] < _reflected_lft_coordinate[0] or _reflected_rgt_coordinate[1] < _reflected_lft_coordinate[1]: # swap if index rgt < index lft
			_reflected_lft_coordinate, _reflected_rgt_coordinate = _reflected_rgt_coordinate, _reflected_lft_coordinate

		return (_reflected_lft_coordinate, _reflected_rgt_coordinate)

	def reflect(self):
		_reflected_board = Board(self.count_rows, self.count_cols)
		
		for _edge in self.__get_edges_coordinates():
			_reflected_edge = self.__reflect_edge(*_edge)
			_reflected_board.edges[self.get_board_position(*_reflected_edge)] = self.edges[self.get_board_position(*_edge)]

		return _reflected_board

	def taken_edges(self):
		return [x for x in self.__get_edges_coordinates() if self.edges[self.get_board_position(*x)]]


class BoardSaver:
	def __init__(self, size):
		self.boards_size = size

	def _add(self, board, value):
		raise Exception()

	def _get(self, board):
		raise Exception()

	def _contains(self, board):
		raise Exception
	
	def equivalent_board(self, state):
		return Board(self.boards_size + 1, self.boards_size + 1, state)

	def contains(self, state):
		"""
		Return whether any equivalent board is contained.
		"""
		return self._contains(self.equivalent_board(state))

	def get(self, state):
		return self._get(self.equivalent_board(state))

	def add(self, state, value):
		"""
		Add a board to the set.

		Parameters:
		board (Board)
		"""
		self._add(self.equivalent_board(state), value)
		return


class BoardHashSaver(BoardSaver):
	"""
	A bit trie set that stores all the boards so as to compare for existence quickly.
	"""
	def __init__(self, size):
		super().__init__(size)
		self.boards = {}
		return

	def _contains(self, board):
		"""
		Parameters:
		board (Board)

		Returns:
		bool indicating whether is included or not. 
		"""
		return board in self.boards

	def _add(self, board, value):
		"""
		Add a board to the set.

		Parameters:
		board (Board)
		"""
		self.boards[board] = value

		return

	def _get(self, board):
		return self.boards[board]


class BoardTrieSaver(BoardSaver):
	class BoardTrieNode:
		def __init__(self):
			self.children = {}
			self.value = None

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

		for _taken_edge in board.get_ordered_edges_values():
			if _taken_edge in root.keys() and is_contained:
				root = root[_taken_edge] # keep traversing the trie
			else:
				is_contained = False
				break

		return is_contained

	def add_board(self, board, value):
		"""
		Add a board to the set.

		Parameters:
		board (Board)
		"""
		if not self.contains_equivalent_board(board): # only add it if it is new
			root = self.bit_trie

			for _taken_edge in board.get_ordered_edges_values():
				if _taken_edge not in root.keys():
					root.children[_taken_edge] = BoardTrieSaver.BoardTrieNode()
				
				root = root.children[_taken_edge]
			root.value = value

		return


class GameStatus:
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
if __name__ == "__main__":
	rows = 3
	cols = 3
	list_of_tuples = [((0,1), (1,1)), ((2,1),(2,2))]
	list_of_tuples_rotated = [((1,0), (2,0)), ((1,1),(1,2))]

	board_saver = BoardHashSaver()
	board = Board(rows, cols, list_of_tuples)
	board_rotated = Board(rows, cols, list_of_tuples_rotated)

	assert(False == board_saver.contains_board(board)) # output False
	assert(False == board_saver.contains_board(board_rotated)) # output False
	assert(False == board_saver.contains_equivalent_board(board)) # output False
	assert(False == board_saver.contains_equivalent_board(board_rotated)) # output False

	board_saver.add_board(board)

	assert(True == board_saver.contains_board(board)) # output True
	assert(False == board_saver.contains_board(board_rotated)) # output False
	assert(True == board_saver.contains_equivalent_board(board)) # output True
	assert(True == board_saver.contains_equivalent_board(board_rotated)) # output True
