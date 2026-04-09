# ============================================================
# src/path_planning/dijkstra.py
# ============================================================
# PURPOSE:
#   Implements Dijkstra's shortest path algorithm.
#   Unlike A*, Dijkstra has no heuristic — it explores ALL
#   cells equally by cost (like BFS with weighted edges).
#
# COMPARISON vs A*:
#   - Dijkstra: Slower (explores more cells), but guaranteed
#     shortest path on any graph.
#   - A*: Faster (uses heuristic), same optimal result on
#     uniform-cost grids.
#
# WHEN TO USE DIJKSTRA:
#   - When heuristic is not available
#   - When edge weights vary (different terrain costs)
#   - When optimality is mandatory
# ============================================================

import heapq
import math
from typing import List, Tuple, Optional, Dict, Set
from src.simulation.grid import Grid, OBSTACLE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DijkstraPlanner:
    """
    Dijkstra's algorithm path planner for 2D grid navigation.

    Supports:
      - Uniform cost (all free cells cost 1.0)
      - Optional diagonal movement (cost sqrt(2))
      - Visited cell tracking for visualization
    """

    def __init__(self, allow_diagonal: bool = False):
        """
        Args:
            allow_diagonal: If True, diagonal moves are allowed.
        """
        self.allow_diagonal = allow_diagonal
        self.visited_cells: List[Tuple[int, int]] = []
        self.explored_count: int = 0

    def find_path(
        self,
        grid: Grid,
        start: Tuple[int, int] = None,
        goal: Tuple[int, int] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Run Dijkstra's algorithm from start to goal.

        Args:
            grid: The Grid object.
            start: (row, col) start. Uses grid.start if None.
            goal:  (row, col) goal.  Uses grid.goal if None.

        Returns:
            List of (row, col) tuples (path), or None if unreachable.
        """
        start = start or grid.start
        goal  = goal  or grid.goal

        if start is None or goal is None:
            logger.error("Start or goal not set.")
            return None
        if start == goal:
            return [start]

        self.visited_cells = []
        self.explored_count = 0

        # dist[cell] = shortest distance from start
        dist: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        closed_set: Set[Tuple[int, int]] = set()

        # Priority queue: (cost, tie_breaker, cell)
        heap: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0
        heapq.heappush(heap, (0.0, counter, start))

        while heap:
            cost, _, current = heapq.heappop(heap)

            if current in closed_set:
                continue
            closed_set.add(current)
            self.visited_cells.append(current)
            self.explored_count += 1

            if current == goal:
                logger.info(
                    f"Dijkstra path found! Cells explored: {self.explored_count}"
                )
                return self._reconstruct(came_from, goal)

            for neighbor, move_cost in self._get_neighbors(grid, current):
                if neighbor in closed_set:
                    continue
                new_cost = cost + move_cost
                if neighbor not in dist or new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    counter += 1
                    came_from[neighbor] = current
                    heapq.heappush(heap, (new_cost, counter, neighbor))

        logger.warning("Dijkstra: No path found.")
        return None

    def _get_neighbors(
        self, grid: Grid, pos: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Return (neighbor, cost) pairs for valid moves."""
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
                    # Prevent corner-cutting
                    if grid.cells[r][nc] != OBSTACLE and grid.cells[nr][c] != OBSTACLE:
                        result.append(((nr, nc), math.sqrt(2)))

        return result

    def _reconstruct(
        self,
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map."""
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def get_stats(self) -> dict:
        """Return statistics from the last search."""
        return {
            "cells_explored": self.explored_count,
            "visited_cells": len(self.visited_cells),
        }
