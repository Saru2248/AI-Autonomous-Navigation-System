# ============================================================
# tests/test_grid.py
# ============================================================
# Unit tests for the Grid simulation environment.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.simulation.grid import Grid, FREE, OBSTACLE, START, GOAL, PATH, VISITED


class TestGridBasic:

    def test_grid_initialized_all_free(self):
        g = Grid(5, 5)
        assert np.all(g.cells == FREE), "Grid should be all FREE on init"

    def test_grid_dimensions(self):
        g = Grid(10, 15)
        assert g.rows == 10
        assert g.cols == 15

    def test_place_obstacle(self):
        g = Grid(5, 5)
        g.place_obstacle(2, 2)
        assert g.cells[2][2] == OBSTACLE

    def test_remove_obstacle(self):
        g = Grid(5, 5)
        g.place_obstacle(2, 2)
        g.remove_obstacle(2, 2)
        assert g.cells[2][2] == FREE

    def test_toggle_obstacle(self):
        g = Grid(5, 5)
        g.toggle_obstacle(1, 1)
        assert g.cells[1][1] == OBSTACLE
        g.toggle_obstacle(1, 1)
        assert g.cells[1][1] == FREE

    def test_set_start(self):
        g = Grid(5, 5)
        g.set_start(0, 0)
        assert g.start == (0, 0)
        assert g.cells[0][0] == START

    def test_set_goal(self):
        g = Grid(5, 5)
        g.set_goal(4, 4)
        assert g.goal == (4, 4)
        assert g.cells[4][4] == GOAL

    def test_start_and_goal_different(self):
        g = Grid(5, 5)
        g.set_start(0, 0)
        g.set_goal(4, 4)
        assert g.start != g.goal

    def test_is_valid(self):
        g = Grid(5, 5)
        assert g.is_valid(0, 0)
        assert g.is_valid(4, 4)
        assert not g.is_valid(-1, 0)
        assert not g.is_valid(0, 5)
        assert not g.is_valid(5, 5)

    def test_is_free(self):
        g = Grid(5, 5)
        g.place_obstacle(1, 1)
        assert g.is_free(0, 0)
        assert not g.is_free(1, 1)

    def test_out_of_bounds_is_not_free(self):
        g = Grid(5, 5)
        assert not g.is_free(-1, 0)
        assert not g.is_free(0, 10)

    def test_reset_clears_everything(self):
        g = Grid(5, 5)
        g.place_obstacle(2, 2)
        g.set_start(0, 0)
        g.set_goal(4, 4)
        g.reset()
        assert np.all(g.cells == FREE)
        assert g.start is None
        assert g.goal is None

    def test_cannot_toggle_start_or_goal(self):
        g = Grid(5, 5)
        g.set_start(0, 0)
        g.set_goal(4, 4)
        g.toggle_obstacle(0, 0)  # Should not change start
        assert g.cells[0][0] == START
        g.toggle_obstacle(4, 4)  # Should not change goal
        assert g.cells[4][4] == GOAL


class TestGridObstacleGeneration:

    def test_random_obstacles_count(self):
        g = Grid(10, 10)
        g.generate_random_obstacles(density=0.2, seed=42)
        obstacles = np.sum(g.cells == OBSTACLE)
        # 20% of 100 = 20 ± a few
        assert obstacles > 0

    def test_random_obstacles_reproducible(self):
        g1 = Grid(10, 10)
        g2 = Grid(10, 10)
        g1.generate_random_obstacles(density=0.2, seed=42)
        g2.generate_random_obstacles(density=0.2, seed=42)
        assert np.array_equal(g1.cells, g2.cells), \
            "Same seed should produce same grid"

    def test_start_not_obstacle_after_random(self):
        g = Grid(10, 10)
        g.set_start(0, 0)
        g.set_goal(9, 9)
        g.generate_random_obstacles(density=0.3, seed=7)
        assert g.cells[0][0] != OBSTACLE, "Start should never be an obstacle"

    def test_goal_not_obstacle_after_random(self):
        g = Grid(10, 10)
        g.set_start(0, 0)
        g.set_goal(9, 9)
        g.generate_random_obstacles(density=0.3, seed=7)
        assert g.cells[9][9] != OBSTACLE, "Goal should never be an obstacle"


class TestGridPathMarking:

    def test_mark_path(self):
        g = Grid(5, 5)
        g.set_start(0, 0)
        g.set_goal(4, 4)
        path = [(0, 0), (0, 1), (0, 2), (4, 4)]
        g.mark_path(path)
        # Interior path cells should be PATH, not start/goal
        assert g.cells[0][1] == PATH
        assert g.cells[0][2] == PATH
        assert g.cells[0][0] == START  # Start unchanged
        assert g.cells[4][4] == GOAL   # Goal unchanged

    def test_clear_path_marks(self):
        g = Grid(5, 5)
        g.mark_path([(1, 1), (1, 2), (1, 3)])
        g.clear_path_marks()
        assert g.cells[1][1] == FREE
        assert g.cells[1][2] == FREE


class TestGridNeighbors:

    def test_neighbors_center(self):
        g = Grid(5, 5)
        neighbors = g.get_neighbors(2, 2)
        # Should have 4 neighbors for center cell
        assert len(neighbors) == 4

    def test_neighbors_corner(self):
        g = Grid(5, 5)
        neighbors = g.get_neighbors(0, 0)
        # Top-left corner: only right and down
        assert len(neighbors) == 2

    def test_neighbors_skip_obstacles(self):
        g = Grid(5, 5)
        g.place_obstacle(1, 2)  # Block one neighbor of (2,2)
        neighbors = g.get_neighbors(2, 2)
        assert (1, 2) not in neighbors

    def test_neighbors_diagonal(self):
        g = Grid(5, 5)
        neighbors = g.get_neighbors(2, 2, allow_diagonal=True)
        # Center with diagonals: 8
        assert len(neighbors) == 8


class TestGridRepr:

    def test_repr_contains_symbols(self):
        g = Grid(3, 3)
        g.set_start(0, 0)
        g.place_obstacle(1, 1)
        r = repr(g)
        assert "S" in r
        assert "#" in r
        assert "." in r
