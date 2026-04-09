# ============================================================
# src/utils/config.py
# ============================================================
# PURPOSE:
#   Central configuration for the entire project.
#   All tunable parameters are defined here so you can
#   change them in ONE place without hunting through code.
# ============================================================

from dataclasses import dataclass


@dataclass
class SimConfig:
    """Simulation environment settings."""
    grid_size: int = 20          # NxN grid
    cell_size: int = 35          # Pixels per grid cell
    obstacle_density: float = 0.2  # 20% of cells = obstacles
    fps: int = 10                # Frames per second
    save_output: bool = False    # Save screenshots to outputs/


@dataclass
class AgentConfig:
    """Agent / robot settings."""
    name: str = "AutoBot-01"
    sensor_range: int = 4       # How far agent can "see" in cells
    allow_diagonal: bool = False  # Allow diagonal movements


@dataclass
class PlannerConfig:
    """Path planner settings."""
    algorithm: str = "astar"    # "astar" or "dijkstra"
    heuristic: str = "manhattan"  # "manhattan" or "euclidean"
    allow_diagonal: bool = False


@dataclass
class VisualizationConfig:
    """Color scheme and UI settings."""
    # Grid colors (RGB)
    COLOR_BACKGROUND: tuple = (15, 15, 30)       # Dark navy
    COLOR_GRID_LINE: tuple = (30, 30, 60)        # Subtle grid lines
    COLOR_FREE: tuple = (25, 25, 50)             # Dark free cell
    COLOR_OBSTACLE: tuple = (220, 50, 50)        # Red obstacle
    COLOR_START: tuple = (50, 220, 100)          # Green start
    COLOR_GOAL: tuple = (255, 200, 0)            # Gold goal
    COLOR_PATH: tuple = (0, 150, 255)            # Blue path
    COLOR_VISITED: tuple = (60, 60, 100)         # Dark visited
    COLOR_AGENT: tuple = (255, 100, 0)           # Orange agent
    COLOR_DANGER: tuple = (255, 50, 50)          # Red danger ring
    COLOR_SENSOR: tuple = (100, 100, 180, 40)    # Transparent sensor


@dataclass
class ProjectConfig:
    """Top-level project configuration."""
    sim: SimConfig = None
    agent: AgentConfig = None
    planner: PlannerConfig = None
    viz: VisualizationConfig = None

    def __post_init__(self):
        if self.sim is None:
            self.sim = SimConfig()
        if self.agent is None:
            self.agent = AgentConfig()
        if self.planner is None:
            self.planner = PlannerConfig()
        if self.viz is None:
            self.viz = VisualizationConfig()


# Default global config — import this in other modules
DEFAULT_CONFIG = ProjectConfig()
