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

	def get_board_position(self, first_coordinate, second_coordinate):
		_second_coordinate_is_on_right = first_coordinate[1] == second_coordinate[1] + 1

		position = first_coordinate[0] * self.count_cols + 2 * first_coordinate[1] # each node has two indexes in the board
		position += 0 if _second_coordinate_is_on_right else 1 # bits are ordered as right and down edges respectively

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
