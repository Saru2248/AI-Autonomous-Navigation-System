# ============================================================
# src/path_planning/astar.py
# ============================================================
# PURPOSE:
#   Implements the A* (A-Star) path planning algorithm.
#   A* is the industry-standard algorithm for robot navigation
#   because it finds the shortest path efficiently using a
#   heuristic to guide the search toward the goal.
#
# HOW IT WORKS:
#   1. Start at the START cell.
#   2. Explore neighbors, calculating f(n) = g(n) + h(n)
#      - g(n): actual cost from start to current cell
#      - h(n): estimated (heuristic) cost to goal
#   3. Always expand the cell with the lowest f(n).
#   4. When GOAL is reached, reconstruct the path.
#
# TIME COMPLEXITY:  O(E log V) where E=edges, V=vertices
# SPACE COMPLEXITY: O(V)
# ============================================================

import heapq
import math
from typing import List, Tuple, Optional, Dict, Set
from src.simulation.grid import Grid, FREE, START, GOAL, OBSTACLE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AStarPlanner:
    """
    A* path planner for 2D grid navigation.

    Supports:
      - Manhattan heuristic (4-directional movement)
      - Euclidean heuristic (8-directional / diagonal movement)
      - Diagonal movement toggle
      - Visited cell tracking for visualization
    """

    def __init__(self, allow_diagonal: bool = False, heuristic: str = "manhattan"):
        """
        Args:
            allow_diagonal: If True, allows diagonal moves.
            heuristic: "manhattan" or "euclidean".
        """
        self.allow_diagonal = allow_diagonal
        self.heuristic = heuristic
        self.visited_cells: List[Tuple[int, int]] = []
        self.explored_count: int = 0

    # ----------------------------------------------------------
    # Heuristic Functions
    # ----------------------------------------------------------

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Estimate cost from cell a to cell b.
        Manhattan: best for 4-directional grids.
        Euclidean: best for diagonal-allowed grids.
        """
        if self.heuristic == "manhattan":
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        elif self.heuristic == "euclidean":
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")

    # ----------------------------------------------------------
    # Core A* Algorithm
    # ----------------------------------------------------------

    def find_path(
        self,
        grid: Grid,
        start: Tuple[int, int] = None,
        goal: Tuple[int, int] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Run A* search from start to goal on the given grid.

        Args:
            grid: The Grid object.
            start: (row, col) start cell. Uses grid.start if None.
            goal: (row, col) goal cell. Uses grid.goal if None.

        Returns:
            List of (row, col) tuples representing the path from
            start to goal (inclusive), or None if no path exists.
        """
        start = start or grid.start
        goal = goal or grid.goal

        # --- Validation ---
        if start is None or goal is None:
            logger.error("Start or goal not set.")
            return None
        if not grid.is_free(*start) and grid.cells[start[0]][start[1]] != 2:
            logger.error(f"Start cell {start} is blocked.")
            return None
        if not grid.is_free(*goal) and grid.cells[goal[0]][goal[1]] != 3:
            logger.error(f"Goal cell {goal} is blocked.")
            return None
        if start == goal:
            return [start]

        self.visited_cells = []
        self.explored_count = 0

        # --- Priority Queue: (f_score, tie_breaker, node) ---
        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0  # Tie-breaker for equal f scores

        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        f_score = g_score[start] + self._heuristic(start, goal)
        heapq.heappush(open_heap, (f_score, counter, start))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        closed_set: Set[Tuple[int, int]] = set()

        # --- Movement cost ---
        # Straight move: 1.0, Diagonal move: sqrt(2) ≈ 1.414
        straight_cost = 1.0
        diagonal_cost = math.sqrt(2)

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            closed_set.add(current)
            self.visited_cells.append(current)
            self.explored_count += 1

            # Goal reached!
            if current == goal:
                logger.info(
                    f"Path found! Length: {len(came_from)}, "
                    f"Cells explored: {self.explored_count}"
                )
                return self._reconstruct_path(came_from, goal)

            # Explore neighbors
            neighbors = self._get_neighbors_with_cost(grid, current)
            for neighbor, move_cost in neighbors:
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))
                    came_from[neighbor] = current

        logger.warning("No path found — goal is unreachable.")
        return None

    def _get_neighbors_with_cost(
        self, grid: Grid, pos: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Get neighbors with their movement costs."""
        r, c = pos
        straight = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        result = []
        for dr, dc in straight:
            nr, nc = r + dr, c + dc
            if grid.is_valid(nr, nc) and grid.cells[nr][nc] != OBSTACLE:
                result.append(((nr, nc), 1.0))

        if self.allow_diagonal:
            for dr, dc in diagonal:
                nr, nc = r + dr, c + dc
                if grid.is_valid(nr, nc) and grid.cells[nr][nc] != OBSTACLE:
                    # Prevent diagonal "cutting corners" through obstacles
                    if grid.cells[r][nc] != OBSTACLE and grid.cells[nr][c] != OBSTACLE:
                        result.append(((nr, nc), math.sqrt(2)))

        return result

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Trace back the path from goal to start."""
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def get_stats(self) -> Dict:
        """Return statistics about the last search run."""
        return {
            "cells_explored": self.explored_count,
            "visited_cells": len(self.visited_cells),
        }
