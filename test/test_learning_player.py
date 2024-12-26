from src.learning_player import BoardSaver, Rotator, Board
from src.dots_boxes import DotsAndBoxesState


class TestBoardSaver:
    def test_same_action(self):
        size = 2
        state = [((0, 1), (1, 1)), ((2, 1), (2, 2))]
        rotated_state = [((1, 0), (2, 0)), ((1, 1), (1, 2))]
        action = ((0, 0), (1, 0))
        action_rotated = ((0, 1), (0, 2))

        bs = BoardSaver(size)
        bstate = DotsAndBoxesState(state, 0)
        bs.define(bstate, action, 1)

        rotated_bstate = DotsAndBoxesState(rotated_state, 0)

        assert bs.contains(bstate)
        assert bs.contains(rotated_bstate)
        assert bs.get(bstate, action) == 1
        assert bs.get(rotated_bstate, action_rotated) == 1

    def test_all_equivalent_boards(self):
        size = 2
        state = [((0, 1), (1, 1)), ((2, 1), (2, 2))]
        action = ((0, 0), (0, 1))
        equivalent_states = [
            # Same board
            (state, action),
            # Reflected
            ([((1, 1), (2, 1)), ((0, 1), (0, 2))], ((2, 0), (2, 1))),
            # Rotated 90 degrees
            ([((1, 0), (1, 1)), ((0, 2), (1, 2))], ((1, 0), (2, 0))),
            # Reflected
            ([((1, 0), (1, 1)), ((1, 2), (2, 2))], ((0, 0), (1, 0))),
            # Rotated 180 degrees
            ([((1, 1), (2, 1)), ((0, 0), (0, 1))], ((2, 1), (2, 2))),
            # # Reflected
            ([((0, 1), (1, 1)), ((2, 0), (2, 1))], ((0, 1), (0, 2))),
            # Rotated 270 degrees
            ([((1, 1), (1, 2)), ((1, 0), (2, 0))], ((0, 2), (1, 2))),
            # Reflected
            ([((1, 1), (1, 2)), ((0, 0), (1, 0))], ((1, 2), (2, 2))),
        ]

        bs = BoardSaver(size)
        bs.define(DotsAndBoxesState(state, 0), action, 1)
        for s, a in equivalent_states:
            assert bs.contains(DotsAndBoxesState(s, 0))
            assert bs.get(DotsAndBoxesState(s, 0), a) == 1


class TestRotator:
    def test_action_rotation(self):
        rotator = Rotator(2)
        action = ((0, 0), (0, 1))
        expected_order = [((1, 0), (2, 0)), ((2, 1), (2, 2)), ((0, 2), (1, 2)), action]
        new_action = action
        for e in expected_order:
            new_action = rotator.rotate_edge(new_action)
            assert new_action == e

    def test_board_rotation(self):
        rotator = Rotator(2)
        state = [
            ((0, 0), (1, 0)),
            ((0, 1), (0, 2)),
            ((0, 2), (1, 2)),
            ((1, 0), (1, 1)),
            ((1, 2), (2, 2)),
            ((2, 0), (2, 1)),
        ]
        b = Board(rotator, 2, state)
        for i, e in enumerate(b.rotations()):
            if i > 0:
                continue

            assert sum(e.edges) == len(state)
            assert len(e.taken_edges()) == len(state)

    def test_reflect_edge(self):
        rotator = Rotator(2)

        cases = [(((0, 1), (1, 1)), ((1, 1), (2, 1))), (((1, 0), (2, 0)), ((0, 0), (1, 0)))]

        for e, r in cases:
            assert rotator.reflect_edge(e) == r
