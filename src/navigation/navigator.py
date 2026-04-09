# ============================================================
# src/navigation/navigator.py
# ============================================================
# PURPOSE:
#   The Navigator orchestrates the full navigation pipeline:
#     1. Receives the grid map and agent state
#     2. Calls the path planner (A* / Dijkstra)
#     3. Validates the path with obstacle detector
#     4. Commands the agent to move step by step
#     5. Handles replanning if the path is blocked
#
# ROLE IN THE SYSTEM:
#   Navigator = BRAIN of the robot
#   It connects Perception → Planning → Action
# ============================================================

from typing import List, Tuple, Optional
from src.simulation.grid import Grid
from src.navigation.agent import Agent, STATUS_REACHED_GOAL, STATUS_NAVIGATING
from src.path_planning.astar import AStarPlanner
from src.path_planning.dijkstra import DijkstraPlanner
from src.path_planning.q_learning import QLearningPlanner
from src.perception.obstacle_detector import ObstacleDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Navigator:
    """
    High-level navigation controller.

    Connects all system modules:
      - Grid (world map)
      - Agent (robot body)
      - Path Planner (A* / Dijkstra)
      - Obstacle Detector (perception)

    Implements replanning: if the planned path becomes blocked
    by a newly detected obstacle, it replans automatically.

    Args:
        algorithm (str): "astar" or "dijkstra"
        sensor_range (int): Agent sensor scan radius
        allow_diagonal (bool): Allow diagonal movement
    """

    def __init__(
        self,
        algorithm: str = "astar",
        sensor_range: int = 4,
        allow_diagonal: bool = False
    ):
        self.algorithm = algorithm
        self.allow_diagonal = allow_diagonal

        # Sub-modules
        self.agent = Agent("AutoBot-01")
        self.detector = ObstacleDetector(sensor_range=sensor_range)

        # Select planner
        if algorithm == "astar":
            self.planner = AStarPlanner(allow_diagonal=allow_diagonal)
        elif algorithm == "dijkstra":
            self.planner = DijkstraPlanner(allow_diagonal=allow_diagonal)
        elif algorithm == "qlearning":
            self.planner = QLearningPlanner(allow_diagonal=allow_diagonal, episodes=1000)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # State tracking
        self.current_path: List[Tuple[int, int]] = []
        self.last_scan: dict = {}
        self.replan_count: int = 0
        self.navigation_log: List[str] = []

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------

    def setup(self, grid: Grid) -> bool:
        """
        Initialize navigation for the given grid.
        Validates start/goal, runs initial path planning.

        Returns:
            True if setup successful and path found, False otherwise.
        """
        if grid.start is None or grid.goal is None:
            logger.error("Grid must have both start and goal set before navigation.")
            return False

        self.agent.initialize(grid.start, grid.goal)

        # Run initial path planning
        path = self.planner.find_path(grid)
        if path is None:
            logger.error("No path found from start to goal!")
            self._log("❌ Navigation setup failed — no path exists.")
            return False

        self.current_path = path
        self.agent.set_path(path)

        # Mark the path on the grid
        grid.mark_path(path)

        self._log(
            f"✅ Navigation ready — {self.algorithm.upper()} path: "
            f"{len(path)} steps, {self.planner.explored_count} cells explored."
        )
        return True

    # ----------------------------------------------------------
    # Core Navigation Step
    # ----------------------------------------------------------

    def step(self, grid: Grid) -> dict:
        """
        Execute one navigation step:
          1. Scan environment with sensor
          2. Check if next cell is safe
          3. Move agent forward
          4. Replan if needed

        Args:
            grid: The current Grid map.

        Returns:
            Step result dictionary with position, status, scan info.
        """
        if self.agent.has_reached_goal:
            return self._make_result("GOAL_REACHED", grid)

        # 1. Perception: scan the environment
        self.last_scan = self.detector.scan(grid, self.agent.position)

        # 2. Check if next step in path is still safe
        if self.agent.path_index < len(self.agent.path):
            next_pos = self.agent.path[self.agent.path_index]
            if not self.detector.is_path_clear(self.agent.position, next_pos, grid):
                logger.warning(f"Path blocked at {next_pos}. Replanning...")
                self._log(f"⚡ Path blocked! Replanning from {self.agent.position}...")
                replanned = self._replan(grid)
                if not replanned:
                    self._log("❌ Replan failed — no path available!")
                    return self._make_result("STUCK", grid)

        # 3. Move agent one step
        new_pos = self.agent.step()

        # 4. Mark agent position on grid (for visualization)
        self._update_grid_marks(grid)

        status = self.agent.status
        self._log(
            f"Step {self.agent.steps}: {self.agent.position} | "
            f"{self.detector.get_scan_summary()}"
        )

        return self._make_result(status, grid)

    # ----------------------------------------------------------
    # Replanning
    # ----------------------------------------------------------

    def _replan(self, grid: Grid) -> bool:
        """
        Replan path from current agent position to goal.

        Returns:
            True if new path found, False otherwise.
        """
        self.replan_count += 1
        new_path = self.planner.find_path(grid, self.agent.position, self.agent.goal)

        if new_path is None:
            return False

        # Clear old path marks and set new path
        grid.clear_path_marks()
        self.current_path = new_path
        self.agent.set_path(new_path)
        grid.mark_path(new_path)

        self._log(
            f"♻️  Replan #{self.replan_count}: new path has {len(new_path)} steps."
        )
        return True

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _update_grid_marks(self, grid: Grid):
        """Update the grid to show current agent position."""
        from src.simulation.grid import AGENT, START, GOAL, PATH
        r, c = self.agent.position
        # Mark agent position
        if grid.cells[r][c] not in (START, GOAL):
            grid.cells[r][c] = AGENT

    def _make_result(self, status: str, grid: Grid) -> dict:
        """Build a result dict from the current step."""
        return {
            "status": status,
            "position": self.agent.position,
            "steps": self.agent.steps,
            "path_remaining": len(self.agent.path) - self.agent.path_index,
            "scan": self.last_scan,
            "agent_stats": self.agent.get_stats(),
            "replan_count": self.replan_count,
        }

    def _log(self, message: str):
        """Append to internal navigation log."""
        self.navigation_log.append(message)
        logger.info(message)

    def get_full_log(self) -> str:
        """Return the full navigation log as a single string."""
        return "\n".join(self.navigation_log)

    def get_summary(self) -> dict:
        """Return final navigation summary."""
        stats = self.agent.get_stats()
        stats["algorithm"] = self.algorithm
        stats["replan_count"] = self.replan_count
        stats["path_length_planned"] = len(self.current_path)
        stats["cells_explored"] = getattr(self.planner, "explored_count", 0)
        return stats
