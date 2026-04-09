# ============================================================
# src/perception/obstacle_detector.py
# ============================================================
# PURPOSE:
#   Simulates the Perception Module of an autonomous robot.
#   In a real system this would use camera/LIDAR/radar.
#   Here we simulate a sensor using a grid-based "sensor field"
#   — the robot scans cells within its detection radius and
#   reports obstacle positions relative to itself.
#
# KEY CONCEPTS:
#   - Sensor Range: How far (in grid cells) the robot can see.
#   - Sensor Sweep: All cells within the sensor range circle.
#   - Obstacle Report: List of detected obstacle positions.
#   - Proximity Warning: Danger flag if obstacle is very close.
# ============================================================

import math
from typing import List, Tuple, Dict
from src.simulation.grid import Grid, OBSTACLE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Proximity danger threshold (cells away from robot)
DANGER_THRESHOLD = 2


class ObstacleDetector:
    """
    Simulated proximity sensor / obstacle detector.

    In a real robot:
      - LIDAR scans 360° and builds a point cloud
      - Camera + CV model detects objects in frame
      - Radar measures distance to obstacles

    Here we simulate this with a radial scan of the grid.

    Attributes:
        sensor_range (int): Detection radius in grid cells.
        detected_obstacles (list): Last detected obstacle positions.
        danger_zone (bool): True if obstacle is within danger threshold.
    """

    def __init__(self, sensor_range: int = 4):
        """
        Args:
            sensor_range: How many grid cells the robot can "see".
        """
        self.sensor_range = sensor_range
        self.detected_obstacles: List[Tuple[int, int]] = []
        self.danger_zone: bool = False
        self.scan_history: List[Dict] = []

    # ----------------------------------------------------------
    # Main Scan Method
    # ----------------------------------------------------------

    def scan(self, grid: Grid, agent_pos: Tuple[int, int]) -> Dict:
        """
        Perform a sensor scan around the agent's current position.

        Args:
            grid: The current Grid map.
            agent_pos: (row, col) of the agent/robot.

        Returns:
            Dictionary with detection results:
            {
                "agent_pos": (r, c),
                "sensor_range": int,
                "detected_obstacles": [(r, c), ...],
                "nearest_obstacle": (r, c) or None,
                "nearest_distance": float or None,
                "danger_zone": bool,
                "cells_scanned": int,
            }
        """
        r, c = agent_pos
        detected = []
        cells_scanned = 0

        # Scan all cells within sensor_range radius
        for dr in range(-self.sensor_range, self.sensor_range + 1):
            for dc in range(-self.sensor_range, self.sensor_range + 1):
                # Use circular scan (Euclidean distance)
                dist = math.sqrt(dr ** 2 + dc ** 2)
                if dist > self.sensor_range:
                    continue  # Outside sensor radius

                nr, nc = r + dr, c + dc
                if not grid.is_valid(nr, nc):
                    continue  # Outside grid bounds

                cells_scanned += 1

                if grid.cells[nr][nc] == OBSTACLE:
                    detected.append((nr, nc))

        # Find the nearest obstacle
        nearest = None
        nearest_dist = None
        if detected:
            distances = [
                (math.sqrt((obs[0] - r) ** 2 + (obs[1] - c) ** 2), obs)
                for obs in detected
            ]
            distances.sort(key=lambda x: x[0])
            nearest_dist, nearest = distances[0]

        # Update danger zone flag
        self.danger_zone = (
            nearest_dist is not None and nearest_dist <= DANGER_THRESHOLD
        )
        self.detected_obstacles = detected

        result = {
            "agent_pos": agent_pos,
            "sensor_range": self.sensor_range,
            "detected_obstacles": detected,
            "nearest_obstacle": nearest,
            "nearest_distance": round(nearest_dist, 2) if nearest_dist else None,
            "danger_zone": self.danger_zone,
            "cells_scanned": cells_scanned,
        }

        self.scan_history.append(result)
        return result

    # ----------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------

    def get_obstacle_directions(
        self, agent_pos: Tuple[int, int]
    ) -> Dict[str, bool]:
        """
        Classify detected obstacles by compass direction.
        Useful for simple reactive avoidance behaviors.

        Returns dict: {"north": bool, "south": bool, "east": bool, "west": bool}
        """
        r, c = agent_pos
        directions = {"north": False, "south": False, "east": False, "west": False}

        for (or_, oc) in self.detected_obstacles:
            if or_ < r:
                directions["north"] = True
            if or_ > r:
                directions["south"] = True
            if oc > c:
                directions["east"] = True
            if oc < c:
                directions["west"] = True

        return directions

    def is_path_clear(
        self,
        agent_pos: Tuple[int, int],
        next_pos: Tuple[int, int],
        grid: Grid
    ) -> bool:
        """
        Check if the direct path from agent to next_pos is free.

        Args:
            agent_pos: Current position.
            next_pos: Intended next position.
            grid: Grid map.

        Returns:
            True if next_pos is obstacle-free.
        """
        r, c = next_pos
        if not grid.is_valid(r, c):
            return False
        if grid.cells[r][c] == OBSTACLE:
            logger.warning(f"Obstacle detected at next position {next_pos}!")
            return False
        return True

    def get_scan_summary(self) -> str:
        """Return a human-readable summary of the last scan."""
        if not self.detected_obstacles:
            return "✅ All clear — no obstacles detected in sensor range."
        msg = (
            f"⚠️  {len(self.detected_obstacles)} obstacle(s) detected. "
            f"Danger zone: {'🚨 YES' if self.danger_zone else 'No'}."
        )
        return msg
