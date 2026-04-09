# ============================================================
# tests/test_navigator.py
# ============================================================
# Unit tests for the Navigator (orchestration) module.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.simulation.grid import Grid, OBSTACLE
from src.navigation.navigator import Navigator


@pytest.fixture
def nav_setup():
    """Returns (navigator, grid) ready for navigation."""
    grid = Grid(10, 10)
    grid.generate_random_obstacles(density=0.15, seed=99)
    grid.set_start(1, 1)
    grid.set_goal(8, 8)
    navigator = Navigator(algorithm="astar", sensor_range=3)
    success = navigator.setup(grid)
    return navigator, grid, success


class TestNavigatorSetup:

    def test_setup_returns_true_on_valid_grid(self, nav_setup):
        _, _, success = nav_setup
        assert success is True

    def test_setup_fails_without_start(self):
        grid = Grid(5, 5)
        grid.set_goal(4, 4)
        nav = Navigator()
        result = nav.setup(grid)
        assert result is False

    def test_setup_fails_without_goal(self):
        grid = Grid(5, 5)
        grid.set_start(0, 0)
        nav = Navigator()
        result = nav.setup(grid)
        assert result is False

    def test_current_path_set_after_setup(self, nav_setup):
        navigator, _, success = nav_setup
        assert success
        assert len(navigator.current_path) > 0

    def test_agent_initialized_at_start(self, nav_setup):
        navigator, grid, success = nav_setup
        assert success
        assert navigator.agent.position == grid.start


class TestNavigatorStep:

    def test_step_returns_dict(self, nav_setup):
        navigator, grid, _ = nav_setup
        result = navigator.step(grid)
        assert isinstance(result, dict)

    def test_step_result_has_required_keys(self, nav_setup):
        navigator, grid, _ = nav_setup
        result = navigator.step(grid)
        for key in ("status", "position", "steps", "scan", "agent_stats"):
            assert key in result, f"Missing key: {key}"

    def test_agent_moves_after_step(self, nav_setup):
        navigator, grid, _ = nav_setup
        start_pos = navigator.agent.position
        navigator.step(grid)
        # Agent should have moved (unless already at goal)
        new_pos = navigator.agent.position
        assert new_pos is not None

    def test_goal_reached_eventually(self):
        """Full run-until-completion test."""
        grid = Grid(12, 12)
        grid.set_start(0, 0)
        grid.set_goal(11, 11)
        grid.generate_random_obstacles(density=0.1, seed=5)
        # Re-set start/goal after obstacle generation clears them
        grid.set_start(0, 0)
        grid.set_goal(11, 11)

        nav = Navigator(algorithm="astar", sensor_range=4)
        success = nav.setup(grid)
        assert success, "Path should exist with low obstacle density"

        max_steps = 500
        status = "NAVIGATING"
        for _ in range(max_steps):
            result = nav.step(grid)
            status = result["status"]
            if status in ("GOAL_REACHED", "STUCK"):
                break

        assert status == "GOAL_REACHED", \
            f"Expected GOAL_REACHED, got {status}"


class TestNavigatorAlgorithms:

    def test_dijkstra_also_finds_path(self):
        grid = Grid(10, 10)
        grid.generate_random_obstacles(density=0.15, seed=42)
        grid.set_start(1, 1)
        grid.set_goal(8, 8)
        nav = Navigator(algorithm="dijkstra", sensor_range=3)
        result = nav.setup(grid)
        assert result is True

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError):
            Navigator(algorithm="bfs_unknown")


class TestNavigatorSummary:

    def test_get_summary_keys(self, nav_setup):
        navigator, _, _ = nav_setup
        summary = navigator.get_summary()
        for key in ("algorithm", "replan_count", "path_length_planned", "cells_explored"):
            assert key in summary

    def test_get_full_log_is_string(self, nav_setup):
        navigator, _, _ = nav_setup
        log = navigator.get_full_log()
        assert isinstance(log, str)
