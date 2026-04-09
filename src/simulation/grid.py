# ============================================================
# src/simulation/grid.py
# ============================================================
# PURPOSE:
#   Defines the Grid — the 2D map/world in which the robot
#   navigates. Each cell is either FREE (0) or OBSTACLE (1).
#   The grid supports random obstacle generation, manual
#   placement, and boundary checking.
# ============================================================

import random
import numpy as np
from typing import List, Tuple, Optional


# Cell type constants
FREE = 0
OBSTACLE = 1
START = 2
GOAL = 3
PATH = 4
VISITED = 5
AGENT = 6


class Grid:
    """
    2D grid map representing the navigation environment.

    Attributes:
        rows (int): Number of rows.
        cols (int): Number of columns.
        cells (np.ndarray): 2D array of cell types.
        start (tuple): (row, col) of start cell.
        goal (tuple): (row, col) of goal cell.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.cells = np.zeros((rows, cols), dtype=int)
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None

    # ----------------------------------------------------------
    # Grid Setup Methods
    # ----------------------------------------------------------

    def place_obstacle(self, row: int, col: int):
        """Mark a single cell as an obstacle."""
        if self.is_valid(row, col):
            self.cells[row][col] = OBSTACLE

    def remove_obstacle(self, row: int, col: int):
        """Remove obstacle from a cell (make it free)."""
        if self.is_valid(row, col):
            if self.cells[row][col] == OBSTACLE:
                self.cells[row][col] = FREE

    def toggle_obstacle(self, row: int, col: int):
        """Toggle obstacle on a cell (free ↔ obstacle)."""
        if self.is_valid(row, col):
            pos = (row, col)
            if pos == self.start or pos == self.goal:
                return  # Don't allow toggling start/goal
            if self.cells[row][col] == OBSTACLE:
                self.cells[row][col] = FREE
            else:
                self.cells[row][col] = OBSTACLE

    def set_start(self, row: int, col: int):
        """Set the start position."""
        if self.is_valid(row, col) and self.cells[row][col] != OBSTACLE:
            # Clear old start
            if self.start:
                r, c = self.start
                self.cells[r][c] = FREE
            self.start = (row, col)
            self.cells[row][col] = START

    def set_goal(self, row: int, col: int):
        """Set the goal position."""
        if self.is_valid(row, col) and self.cells[row][col] != OBSTACLE:
            # Clear old goal
            if self.goal:
                r, c = self.goal
                self.cells[r][c] = FREE
            self.goal = (row, col)
            self.cells[row][col] = GOAL

    def generate_random_obstacles(self, density: float = 0.2, seed: int = None):
        """
        Randomly place obstacles across the grid.

        Args:
            density (float): Fraction of cells to make obstacles (0.0 – 0.5).
            seed (int): Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reset grid to free
        self.cells[:] = FREE

        total_cells = self.rows * self.cols
        num_obstacles = int(total_cells * density)
        placed = 0
        attempts = 0

        while placed < num_obstacles and attempts < total_cells * 10:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            pos = (r, c)
            if self.cells[r][c] == FREE and pos != self.start and pos != self.goal:
                self.cells[r][c] = OBSTACLE
                placed += 1
            attempts += 1

        # Re-mark start and goal if they exist
        if self.start:
            r, c = self.start
            self.cells[r][c] = START
        if self.goal:
            r, c = self.goal
            self.cells[r][c] = GOAL

    def clear_path_marks(self):
        """Remove PATH and VISITED marks but keep obstacles, start, goal."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c] in (PATH, VISITED, AGENT):
                    self.cells[r][c] = FREE

    def reset(self):
        """Reset entire grid to all-free."""
        self.cells[:] = FREE
        self.start = None
        self.goal = None

    # ----------------------------------------------------------
    # Query Methods
    # ----------------------------------------------------------

    def is_valid(self, row: int, col: int) -> bool:
        """Check if (row, col) is within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_free(self, row: int, col: int) -> bool:
        """Check if a cell is navigable (not an obstacle, not out of bounds)."""
        return self.is_valid(row, col) and self.cells[row][col] != OBSTACLE

    def get_neighbors(self, row: int, col: int,
                      allow_diagonal: bool = False) -> List[Tuple[int, int]]:
        """
        Return all valid, free neighboring cells.

        Args:
            row, col: Current cell position.
            allow_diagonal: If True, include diagonal neighbors.

        Returns:
            List of (row, col) tuples of free neighbors.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        if allow_diagonal:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        neighbors = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if self.is_free(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def mark_path(self, path: List[Tuple[int, int]]):
        """Mark cells along a path (excludes start and goal cells)."""
        for r, c in path:
            if self.cells[r][c] not in (START, GOAL):
                self.cells[r][c] = PATH

    def mark_visited(self, row: int, col: int):
        """Mark a cell as visited during search."""
        if self.cells[row][col] not in (START, GOAL, OBSTACLE, PATH):
            self.cells[row][col] = VISITED

    def to_numpy(self) -> np.ndarray:
        """Return the grid as a NumPy array (copy)."""
        return self.cells.copy()

    def __repr__(self):
        symbols = {FREE: ".", OBSTACLE: "#", START: "S", GOAL: "G",
                   PATH: "*", VISITED: "~", AGENT: "A"}
        rows = []
        for r in range(self.rows):
            row_str = " ".join(symbols.get(self.cells[r][c], "?")
                               for c in range(self.cols))
            rows.append(row_str)
        return "\n".join(rows)
