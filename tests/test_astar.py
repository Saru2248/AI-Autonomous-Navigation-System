# ============================================================
# tests/test_astar.py
# ============================================================
# Unit tests for the A* path planning algorithm.
# Run: python -m pytest tests/ -v
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.simulation.grid import Grid, OBSTACLE, FREE
from src.path_planning.astar import AStarPlanner


# ── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def small_grid():
    """5x5 empty grid with start (0,0) and goal (4,4)."""
    g = Grid(5, 5)
    g.set_start(0, 0)
    g.set_goal(4, 4)
    return g


@pytest.fixture
def blocked_grid():
    """5x5 grid where goal is completely surrounded by obstacles."""
    g = Grid(5, 5)
    g.set_start(0, 0)
    g.set_goal(4, 4)
    # Surround goal
    for pos in [(3, 4), (4, 3), (3, 3)]:
        g.place_obstacle(*pos)
    return g


# ── Basic Tests ────────────────────────────────────────────

class TestAStarBasic:

    def test_finds_path_empty_grid(self, small_grid):
        """A* should find a path on an empty grid."""
        planner = AStarPlanner()
        path = planner.find_path(small_grid)
        assert path is not None, "Expected a path, got None"
        assert len(path) > 0

    def test_path_starts_at_start(self, small_grid):
        """Path must begin at the start cell."""
        planner = AStarPlanner()
        path = planner.find_path(small_grid)
        assert path[0] == small_grid.start

    def test_path_ends_at_goal(self, small_grid):
        """Path must end at the goal cell."""
        planner = AStarPlanner()
        path = planner.find_path(small_grid)
        assert path[-1] == small_grid.goal

    def test_no_path_when_blocked(self, blocked_grid):
        """A* returns None when goal is unreachable."""
        planner = AStarPlanner()
        path = planner.find_path(blocked_grid)
        assert path is None, "Expected None when goal blocked"

    def test_path_continuity(self, small_grid):
        """Each step in the path must be adjacent to the previous."""
        planner = AStarPlanner()
        path = planner.find_path(small_grid)
        for i in range(1, len(path)):
            r1, c1 = path[i - 1]
            r2, c2 = path[i]
            # Manhattan distance between consecutive steps must be 1
            assert abs(r1 - r2) + abs(c1 - c2) == 1, \
                f"Non-adjacent steps: {path[i-1]} → {path[i]}"

    def test_path_avoids_obstacles(self):
        """Path must not pass through obstacle cells."""
        g = Grid(7, 7)
        g.set_start(0, 0)
        g.set_goal(6, 6)
        # Place a horizontal wall, gap at bottom
        for c in range(6):
            g.place_obstacle(3, c)
        planner = AStarPlanner()
        path = planner.find_path(g)
        assert path is not None
        obstacle_cells = {
            (r, c) for r in range(g.rows) for c in range(g.cols)
            if g.cells[r][c] == OBSTACLE
        }
        for cell in path:
            assert cell not in obstacle_cells, f"Path goes through obstacle: {cell}"

    def test_start_equals_goal(self):
        """When start == goal, path should be [start]."""
        g = Grid(5, 5)
        g.set_start(2, 2)
        g.set_goal(2, 2)
        planner = AStarPlanner()
        path = planner.find_path(g)
        assert path == [(2, 2)]


# ── Diagonal Tests ─────────────────────────────────────────

class TestAStarDiagonal:

    def test_diagonal_path_shorter(self):
        """Diagonal path should be shorter than Manhattan path."""
        g = Grid(10, 10)
        g.set_start(0, 0)
        g.set_goal(9, 9)

        planner_straight = AStarPlanner(allow_diagonal=False)
        planner_diagonal = AStarPlanner(allow_diagonal=True)

        path_s = planner_straight.find_path(g)
        path_d = planner_diagonal.find_path(g)

        # Diagonal should reach goal in fewer steps
        assert len(path_d) <= len(path_s)

    def test_diagonal_path_valid(self):
        """Diagonal path steps must be at most sqrt(2) apart."""
        import math
        g = Grid(10, 10)
        g.set_start(0, 0)
        g.set_goal(9, 9)
        planner = AStarPlanner(allow_diagonal=True)
        path = planner.find_path(g)
        assert path is not None
        for i in range(1, len(path)):
            r1, c1 = path[i - 1]
            r2, c2 = path[i]
            dist = math.sqrt((r1-r2)**2 + (c1-c2)**2)
            assert dist <= math.sqrt(2) + 1e-6, \
                f"Step too large: {dist} between {path[i-1]} and {path[i]}"


# ── Stats Tests ────────────────────────────────────────────

class TestAStarStats:

    def test_explored_count_positive(self, small_grid):
        """After search, explored_count must be > 0."""
        planner = AStarPlanner()
        planner.find_path(small_grid)
        assert planner.explored_count > 0

    def test_get_stats_keys(self, small_grid):
        """get_stats() must return required keys."""
        planner = AStarPlanner()
        planner.find_path(small_grid)
        stats = planner.get_stats()
        assert "cells_explored" in stats
        assert "visited_cells" in stats


# ── Heuristic Tests ────────────────────────────────────────

class TestAStarHeuristics:

    def test_manhattan_heuristic(self, small_grid):
        """Test with Manhattan heuristic (default)."""
        planner = AStarPlanner(heuristic="manhattan")
        path = planner.find_path(small_grid)
        assert path is not None

    def test_euclidean_heuristic(self, small_grid):
        """Test with Euclidean heuristic."""
        planner = AStarPlanner(heuristic="euclidean")
        path = planner.find_path(small_grid)
        assert path is not None

    def test_invalid_heuristic_raises(self, small_grid):
        """Invalid heuristic must raise ValueError."""
        planner = AStarPlanner(heuristic="invalid")
        with pytest.raises(ValueError):
            planner.find_path(small_grid)
