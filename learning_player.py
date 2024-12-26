import bitstring as bitstring
import itertools


class Rotator:

    def __init__(self, size):
        self.size = size

    def __reflect_coordinate(self, coordinate):
        return self.size - 1 - coordinate[0], coordinate[1]

    def reflect_edge(self, edge):
        lft_coordinate, rgt_coordinate = edge
        _reflected_lft_coordinate = self.__reflect_coordinate(lft_coordinate)
        _reflected_rgt_coordinate = self.__reflect_coordinate(rgt_coordinate)

        if (
            _reflected_rgt_coordinate[0] < _reflected_lft_coordinate[0]
            or _reflected_rgt_coordinate[1] < _reflected_lft_coordinate[1]
        ):  # swap if index rgt < index lft
            _reflected_lft_coordinate, _reflected_rgt_coordinate = _reflected_rgt_coordinate, _reflected_lft_coordinate

        return _reflected_lft_coordinate, _reflected_rgt_coordinate

    def rotate_coordinate(self, coordinate):
        return -coordinate[1] + self.size - 1, coordinate[0]

    def rotate_edge(self, edge):
        lft_coordinate, rgt_coordinate = edge
        _next_lft_coordinate = self.rotate_coordinate(lft_coordinate)
        _next_rgt_coordinate = self.rotate_coordinate(rgt_coordinate)

        if (
            _next_rgt_coordinate[0] < _next_lft_coordinate[0] or _next_rgt_coordinate[1] < _next_lft_coordinate[1]
        ):  # swap
            _next_lft_coordinate, _next_rgt_coordinate = _next_rgt_coordinate, _next_lft_coordinate

        return _next_lft_coordinate, _next_rgt_coordinate


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

    def __init__(self, rotator, size, taken_edges=()):
        self.size = size
        self.edges = self.clean_board()
        self._update_taken_edges(taken_edges)
        self.rotator = rotator
        return

    def __hash__(self):
        mask = 0
        for _ith, _bool in enumerate(self.edges):
            mask |= _bool << _ith
        return mask

    def __eq__(self, other_board):
        return self.size == other_board.size and self.edges == other_board.edges

    def clean_board(self):
        """
        Return a new default board game representation with no taken edge.
        """
        return bitstring.BitArray(uint=0, length=self.size * self.size * 2)

    def __check_coordinate_bound(self, coordinate):
        """
        Check coordinate is inside rows/cols board bounds.
        """
        if self.size <= coordinate[0] or self.size <= coordinate[1]:
            raise Exception("Invalid coordinate {} for size {}".format(coordinate, self.size))
        return

    def __check_existing_edge(self, first_coordinate, second_coordinate):
        """
        Check there is an edge between coordinates
        """
        if first_coordinate == second_coordinate:
            raise Exception(
                "first_coordinate {} equals second_coordinate {}".format(first_coordinate, second_coordinate)
            )

        if second_coordinate[0] < first_coordinate[0] or second_coordinate[1] < first_coordinate[1]:
            raise Exception(
                "first_coordinate {} is not on top or left of second_coordinate {}".format(
                    first_coordinate, second_coordinate
                )
            )

        if first_coordinate[0] not in range(second_coordinate[0] - 1, second_coordinate[0] + 1) or first_coordinate[
            1
        ] not in range(second_coordinate[1] - 1, second_coordinate[1] + 1):
            raise Exception(
                "first_coordinate {} is contiguous to second_coordinate {}".format(first_coordinate, second_coordinate)
            )
        return

    def get_board_position(self, first_coordinate, second_coordinate):
        _share_row = first_coordinate[1] + 1 == second_coordinate[1]

        position = 2 * (
            first_coordinate[0] * self.size + first_coordinate[1]
        )  # each coordinate has two bools in the board
        position += 0 if _share_row else 1  # its bool for same row comes before the bool for same column

        # only to check errors during development
        for _coordinate in [first_coordinate, second_coordinate]:
            self.__check_coordinate_bound(_coordinate)

        self.__check_existing_edge(first_coordinate, second_coordinate)

        return position

    def _update_taken_edges(self, taken_edges):
        """
        taken_edges :list: of edges, where and edge is a pair of coordinates and a coordinate a pair of int
        """
        for _first_coordinate, _second_coordinate in taken_edges:
            self.edges[self.get_board_position(_first_coordinate, _second_coordinate)] = 1
        return

    def __get_edges_coordinates(self):
        for _coordinate in itertools.product(range(self.size - 1), range(self.size - 1)):  # inner edges
            yield (_coordinate, (_coordinate[0] + 1, _coordinate[1]))
            yield (_coordinate, (_coordinate[0], _coordinate[1] + 1))

        for _row in range(self.size - 1):  # last row
            yield ((_row, self.size - 1), (_row + 1, self.size - 1))

        for _col in range(self.size - 1):  # last column
            yield ((self.size - 1, _col), (self.size - 1, _col + 1))

    def rotations(self):
        yield self
        yield self.reflect()

        _prev_rotated_board = self

        for _ in range(3):  # just rate
            _rotate_board = Board(self.rotator, self.size)

            for _actual_edge in self.__get_edges_coordinates():
                _rotated_edge = self.rotator.rotate_edge(_actual_edge)
                _rotate_board.edges[self.get_board_position(*_rotated_edge)] = _prev_rotated_board.edges[
                    self.get_board_position(*_actual_edge)
                ]

            yield _rotate_board
            yield _rotate_board.reflect()

            _prev_rotated_board = _rotate_board

        return

    def reflect(self):
        _reflected_board = Board(self.rotator, self.size)

        for _edge in self.__get_edges_coordinates():
            _reflected_edge = self.rotator.reflect_edge(_edge)
            _reflected_board.edges[self.get_board_position(*_reflected_edge)] = self.edges[
                self.get_board_position(*_edge)
            ]

        return _reflected_board

    def taken_edges(self):
        return [x for x in self.__get_edges_coordinates() if self.edges[self.get_board_position(*x)]]


class Action:

    def __init__(self, rotator, edge):
        self.edge = edge
        self.rotator = rotator

    def __hash__(self):
        return self.edge.__hash__()

    def __eq__(self, other_action):
        inverted = (self.edge[1], self.edge[0])
        return self.edge == other_action.edge or inverted == other_action.edge

    def reflect(self):
        return Action(self.rotator, self.rotator.reflect_edge(self.edge))

    def rotations(self):
        yield self
        yield self.reflect()

        _rotated_edge = self.edge

        for _ in range(3):  # just rate
            _rotated_edge = self.rotator.rotate_edge(_rotated_edge)
            _rotated_action = Action(self.rotator, _rotated_edge)
            yield _rotated_action
            yield _rotated_action.reflect()

        return


class BoardSaver:

    def __init__(self, size):
        self.size = size
        self.boards = {}
        self.rotator = Rotator(self.size)

    def _equivalent_board(self, state):
        _board = Board(self.rotator, self.size + 1, state)
        return min(_board.rotations(), key=lambda x: x.__hash__())

    def _equivalent_board_action(self, state, action):
        _board = Board(self.rotator, self.size + 1, state)
        _action = Action(self.rotator, action)
        return min(zip(_board.rotations(), _action.rotations()), key=lambda x: x[0].__hash__())

    def contains(self, state):
        """
        Return whether any equivalent board is contained.
        """
        _board = self._equivalent_board(state)
        return _board in self.boards

    def _contains(self, state, action):
        _board, _action = self._equivalent_board_action(state, action)

        return _action in self.boards[_board]

    def get(self, state, action):
        _board, _action = self._equivalent_board_action(state, action)

        return self.boards[_board][_action]

    def _get(self, state):
        return self.boards[self._equivalent_board(state)]

    def define(self, state, action, value):
        """
        Add a board to the set.

        Parameters:
        board
        action
        value
        """
        _board, _action = self._equivalent_board_action(state, action)
        if _board not in self.boards:
            self.boards[_board] = {}

        self.boards[_board][_action] = value
        return


# example
if __name__ == "__main__":
    size = 3
    list_of_tuples = [((0, 1), (1, 1)), ((2, 1), (2, 2))]
    list_of_tuples_rotated = [((1, 0), (2, 0)), ((1, 1), (1, 2))]
    action = ((0, 0), (1, 0))
    action_rotated = ((0, 1), (0, 2))

    bs = BoardSaver(size)
    bs.define(list_of_tuples, action, 1)

    assert bs.contains(list_of_tuples)
    assert bs.get(list_of_tuples, action) == 1
    assert bs.get(list_of_tuples_rotated, action_rotated) == 1
