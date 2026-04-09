# ============================================================
# src/navigation/agent.py
# ============================================================
# PURPOSE:
#   Represents the autonomous robot/agent in the simulation.
#   Tracks its position, heading, path, and movement history.
#
# ANALOGY:
#   In a self-driving car: the vehicle body + odometer.
#   In a warehouse robot: the physical robot frame + encoder.
# ============================================================

import math
from typing import List, Tuple, Optional

# Agent status codes
STATUS_IDLE          = "IDLE"
STATUS_NAVIGATING    = "NAVIGATING"
STATUS_REACHED_GOAL  = "GOAL_REACHED"
STATUS_STUCK         = "STUCK"

# Compass headings
HEADING_MAP = {
    (-1,  0): "North",
    ( 1,  0): "South",
    ( 0,  1): "East",
    ( 0, -1): "West",
    (-1, -1): "NW",
    (-1,  1): "NE",
    ( 1, -1): "SW",
    ( 1,  1): "SE",
}


class Agent:
    """
    Autonomous navigation agent.

    Attributes:
        name (str):                 Agent identifier.
        position (tuple):           Current (row, col).
        goal (tuple):               Target (row, col).
        path (list):                Planned waypoint list.
        path_index (int):           Current position in path.
        steps (int):                Total steps taken.
        total_distance (float):     Cumulative distance traveled.
        heading (str):              Current compass direction.
        status (str):               Navigation status code.
        position_history (list):    All visited positions.
    """

    def __init__(self, name: str = "Agent-01"):
        self.name = name
        self.position: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        self.path: List[Tuple[int, int]] = []
        self.path_index: int = 0
        self.steps: int = 0
        self.total_distance: float = 0.0
        self.heading: str = "North"
        self.status: str = STATUS_IDLE
        self.position_history: List[Tuple[int, int]] = []
        self.has_reached_goal: bool = False

    # ----------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------

    def initialize(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ):
        """
        Set starting position and goal.

        Args:
            start: (row, col) starting cell.
            goal:  (row, col) target cell.
        """
        self.position = start
        self.goal = goal
        self.path = []
        self.path_index = 0
        self.steps = 0
        self.total_distance = 0.0
        self.heading = "North"
        self.status = STATUS_IDLE
        self.position_history = [start]
        self.has_reached_goal = False

    def set_path(self, path: List[Tuple[int, int]]):
        """
        Assign a new planned path to follow.

        Args:
            path: List of (row, col) from start to goal.
        """
        self.path = path
        # Start from index 1 (skip current position which is path[0])
        self.path_index = 1 if len(path) > 1 else 0
        self.status = STATUS_NAVIGATING

    # ----------------------------------------------------------
    # Movement
    # ----------------------------------------------------------

    def step(self) -> Optional[Tuple[int, int]]:
        """
        Move agent one step along the planned path.

        Returns:
            New position, or None if already at goal or no path.
        """
        if self.position == self.goal:
            self.status = STATUS_REACHED_GOAL
            self.has_reached_goal = True
            return self.position

        if self.path_index >= len(self.path):
            # No more waypoints
            if self.position == self.goal:
                self.status = STATUS_REACHED_GOAL
                self.has_reached_goal = True
            else:
                self.status = STATUS_STUCK
            return self.position

        # Move to next waypoint
        prev_pos = self.position
        new_pos = self.path[self.path_index]
        self.path_index += 1

        # Update tracking
        dist = self._euclidean(prev_pos, new_pos)
        self.total_distance += dist
        self.heading = self._compute_heading(prev_pos, new_pos)
        self.position = new_pos
        self.steps += 1
        self.position_history.append(new_pos)

        # Check if goal reached
        if new_pos == self.goal:
            self.status = STATUS_REACHED_GOAL
            self.has_reached_goal = True

        return new_pos

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    @staticmethod
    def _euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance between two grid cells."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def _compute_heading(
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int]
    ) -> str:
        """Convert movement delta to compass direction."""
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        return HEADING_MAP.get((dr, dc), "Unknown")

    def get_stats(self) -> dict:
        """Return current agent statistics dictionary."""
        return {
            "name": self.name,
            "position": self.position,
            "goal": self.goal,
            "steps": self.steps,
            "total_distance": round(self.total_distance, 2),
            "heading": self.heading,
            "status": self.status,
            "path_remaining": max(0, len(self.path) - self.path_index),
            "history_length": len(self.position_history),
        }

    def __repr__(self) -> str:
        return (
            f"Agent({self.name!r}, pos={self.position}, "
            f"steps={self.steps}, status={self.status!r})"
        )
