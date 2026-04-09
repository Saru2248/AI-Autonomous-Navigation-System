# ============================================================
# tests/test_perception.py
# ============================================================
# Unit tests for the Perception / ObstacleDetector module.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.simulation.grid import Grid, OBSTACLE
from src.perception.obstacle_detector import ObstacleDetector


@pytest.fixture
def grid_with_obstacles():
    """10x10 grid with known obstacle at (5,5)."""
    g = Grid(10, 10)
    g.set_start(0, 0)
    g.set_goal(9, 9)
    g.place_obstacle(5, 5)
    g.place_obstacle(3, 3)
    return g


class TestObstacleDetectorScan:

    def test_scan_returns_dict(self, grid_with_obstacles):
        det = ObstacleDetector(sensor_range=4)
        result = det.scan(grid_with_obstacles, (4, 4))
        assert isinstance(result, dict)

    def test_scan_required_keys(self, grid_with_obstacles):
        det = ObstacleDetector(sensor_range=4)
        result = det.scan(grid_with_obstacles, (4, 4))
        for key in ("agent_pos", "sensor_range", "detected_obstacles",
                    "nearest_obstacle", "nearest_distance", "danger_zone",
                    "cells_scanned"):
            assert key in result, f"Missing key: {key}"

    def test_detects_nearby_obstacle(self, grid_with_obstacles):
        """Obstacle at (3,3) should be detected from (4,4) with range 4."""
        det = ObstacleDetector(sensor_range=4)
        result = det.scan(grid_with_obstacles, (4, 4))
        assert (3, 3) in result["detected_obstacles"]

    def test_does_not_detect_far_obstacle(self):
        """Obstacle at (0,0) must not be detected from (9,9) with range 2."""
        g = Grid(10, 10)
        g.place_obstacle(0, 0)
        det = ObstacleDetector(sensor_range=2)
        result = det.scan(g, (9, 9))
        assert (0, 0) not in result["detected_obstacles"]

    def test_no_obstacles_clear(self):
        """Empty grid → no obstacles detected."""
        g = Grid(10, 10)
        det = ObstacleDetector(sensor_range=5)
        result = det.scan(g, (5, 5))
        assert result["detected_obstacles"] == []
        assert result["danger_zone"] is False
        assert result["nearest_obstacle"] is None

    def test_danger_zone_triggered_close(self):
        """Obstacle 1 cell away → danger_zone=True."""
        g = Grid(10, 10)
        g.place_obstacle(5, 6)  # 1 cell to the right of (5,5)
        det = ObstacleDetector(sensor_range=5)
        result = det.scan(g, (5, 5))
        assert result["danger_zone"] is True, "Obstacle 1 cell away should trigger danger"

    def test_danger_zone_clear_far(self):
        """Obstacle 4 cells away → danger_zone=False (threshold=2)."""
        g = Grid(10, 10)
        g.place_obstacle(5, 9)  # 4 cells from (5,5)
        det = ObstacleDetector(sensor_range=5)
        result = det.scan(g, (5, 5))
        assert result["danger_zone"] is False

    def test_cells_scanned_positive(self, grid_with_obstacles):
        det = ObstacleDetector(sensor_range=3)
        result = det.scan(grid_with_obstacles, (5, 5))
        assert result["cells_scanned"] > 0


class TestObstacleDetectorHelpers:

    def test_is_path_clear_free(self):
        g = Grid(5, 5)
        det = ObstacleDetector()
        assert det.is_path_clear((2, 2), (2, 3), g) is True

    def test_is_path_clear_blocked(self):
        g = Grid(5, 5)
        g.place_obstacle(2, 3)
        det = ObstacleDetector()
        assert det.is_path_clear((2, 2), (2, 3), g) is False

    def test_is_path_clear_out_of_bounds(self):
        g = Grid(5, 5)
        det = ObstacleDetector()
        assert det.is_path_clear((0, 0), (-1, 0), g) is False

    def test_get_obstacle_directions(self):
        g = Grid(10, 10)
        g.place_obstacle(3, 5)  # North of (5,5)
        g.place_obstacle(7, 5)  # South of (5,5)
        det = ObstacleDetector(sensor_range=5)
        det.scan(g, (5, 5))
        dirs = det.get_obstacle_directions((5, 5))
        assert dirs["north"] is True
        assert dirs["south"] is True
        assert dirs["east"] is False
        assert dirs["west"] is False

    def test_scan_summary_no_obstacles(self):
        g = Grid(5, 5)
        det = ObstacleDetector()
        det.scan(g, (2, 2))
        summary = det.get_scan_summary()
        assert "clear" in summary.lower() or "no obstacle" in summary.lower()

    def test_scan_summary_with_obstacles(self):
        g = Grid(5, 5)
        g.place_obstacle(2, 3)
        det = ObstacleDetector(sensor_range=3)
        det.scan(g, (2, 2))
        summary = det.get_scan_summary()
        assert "obstacle" in summary.lower()

    def test_scan_history_grows(self):
        g = Grid(5, 5)
        det = ObstacleDetector()
        for _ in range(5):
            det.scan(g, (2, 2))
        assert len(det.scan_history) == 5
