# ============================================================
# src/simulation/environment.py
# ============================================================
# PURPOSE:
#   The main Pygame simulation window and event loop.
#   Renders the grid, agent, obstacles, path, and HUD.
#   Handles all user interactions (mouse clicks, keyboard).
#
# CONTROLS (shown in HUD):
#   Left Click          — Toggle obstacle on/off
#   Right Click         — Set START (first click) / GOAL (second click)
#   SPACE               — Run path planning + start navigation
#   R                   — Reset grid (random obstacles)
#   C                   — Clear all obstacles
#   S                   — Save screenshot
#   ESC / Q             — Quit
#   Tab                 — Switch algorithm (A* / Dijkstra)
# ============================================================

import pygame
import sys
import os
import random
from typing import Tuple, Optional

from src.simulation.grid import (
    Grid, FREE, OBSTACLE, START, GOAL, PATH, VISITED, AGENT
)
from src.navigation.navigator import Navigator
from src.utils.config import DEFAULT_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Pygame colors ──────────────────────────────────────────
C = DEFAULT_CONFIG.viz
BG_COLOR       = C.COLOR_BACKGROUND
GRID_LINE      = C.COLOR_GRID_LINE
FREE_COLOR     = C.COLOR_FREE
OBSTACLE_COLOR = C.COLOR_OBSTACLE
START_COLOR    = C.COLOR_START
GOAL_COLOR     = C.COLOR_GOAL
PATH_COLOR     = C.COLOR_PATH
VISITED_COLOR  = C.COLOR_VISITED
AGENT_COLOR    = C.COLOR_AGENT

CELL_TYPE_COLORS = {
    FREE:     FREE_COLOR,
    OBSTACLE: OBSTACLE_COLOR,
    START:    START_COLOR,
    GOAL:     GOAL_COLOR,
    PATH:     PATH_COLOR,
    VISITED:  VISITED_COLOR,
    AGENT:    AGENT_COLOR,
}

# HUD Panel width (right side info panel)
HUD_WIDTH = 280


class SimulationEnvironment:
    """
    Main pygame-based 2D grid simulation.

    Features:
      - Interactive obstacle placement
      - Real-time A* / Dijkstra path visualization
      - Live agent animation moving along path
      - HUD with stats, controls, and status
      - Screenshot save support
    """

    def __init__(
        self,
        grid_size: int = 20,
        cell_size: int = 35,
        obstacle_density: float = 0.2,
        fps: int = 10,
        save_output: bool = False,
        algorithm: str = "astar"
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.obstacle_density = obstacle_density
        self.fps = fps
        self.save_output = save_output
        self.algorithm = algorithm

        # Computed dimensions
        self.grid_px = grid_size * cell_size         # Grid pixel area
        self.win_w = self.grid_px + HUD_WIDTH
        self.win_h = self.grid_px

        # Create grid model
        self.grid = Grid(grid_size, grid_size)
        self.grid.generate_random_obstacles(density=obstacle_density, seed=42)

        # Default start and goal (top-left → bottom-right)
        self.grid.set_start(1, 1)
        self.grid.set_goal(grid_size - 2, grid_size - 2)

        # Navigator
        self.navigator = Navigator(algorithm=self.algorithm, sensor_range=4)
        self.nav_ready = False
        self.nav_running = False
        self.nav_done = False
        self.step_result = {}

        # Interaction state
        self.placement_mode = "obstacle"  # "obstacle" | "start" | "goal"
        self.status_msg = "Press SPACE to plan & navigate | Click to toggle obstacles"
        self.algo_names = ["astar", "dijkstra", "qlearning"]
        self.algo_idx = 0

        # Pygame init
        pygame.init()
        pygame.display.set_caption("AI Autonomous Navigation System")
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title  = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_body   = pygame.font.SysFont("Consolas", 13)
        self.font_small  = pygame.font.SysFont("Consolas", 11)
        self.font_large  = pygame.font.SysFont("Consolas", 20, bold=True)

        logger.info("SimulationEnvironment initialized.")

    # ──────────────────────────────────────────────────────
    # Main Loop
    # ──────────────────────────────────────────────────────

    def run(self):
        """Main game loop."""
        logger.info("Starting simulation loop. Press SPACE to begin navigation.")
        running = True

        while running:
            self.clock.tick(self.fps)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keydown(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event)

            # Navigation step
            if self.nav_running and not self.nav_done:
                result = self.navigator.step(self.grid)
                self.step_result = result
                if result["status"] in ("GOAL_REACHED", "STUCK"):
                    self.nav_done = True
                    self.nav_running = False
                    msg = (
                        "🎯 Goal Reached!" if result["status"] == "GOAL_REACHED"
                        else "❌ Agent STUCK — no path!"
                    )
                    self.status_msg = msg
                    logger.info(msg)
                    if self.save_output:
                        self._save_screenshot()

            # Draw everything
            self._draw()
            pygame.display.flip()

        self._quit()

    # ──────────────────────────────────────────────────────
    # Event Handlers
    # ──────────────────────────────────────────────────────

    def _handle_keydown(self, event) -> bool:
        """Handle keyboard input. Returns False to quit."""
        key = event.key

        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False  # Quit

        elif key == pygame.K_SPACE:
            self._start_navigation()

        elif key == pygame.K_r:
            self._reset_random()

        elif key == pygame.K_c:
            self._clear_obstacles()

        elif key == pygame.K_s:
            self._save_screenshot()

        elif key == pygame.K_TAB:
            self._toggle_algorithm()

        elif key == pygame.K_F1:
            self.placement_mode = "obstacle"
            self.status_msg = "Mode: Obstacle placement"

        elif key == pygame.K_F2:
            self.placement_mode = "start"
            self.status_msg = "Mode: Click to set START"

        elif key == pygame.K_F3:
            self.placement_mode = "goal"
            self.status_msg = "Mode: Click to set GOAL"

        return True

    def _handle_click(self, event):
        """Handle mouse clicks on the grid."""
        x, y = event.pos

        # Only process clicks on the grid area
        if x >= self.grid_px:
            return

        col = x // self.cell_size
        row = y // self.cell_size

        if not self.grid.is_valid(row, col):
            return

        if event.button == 1:  # Left click
            if self.placement_mode == "obstacle":
                self.grid.toggle_obstacle(row, col)
                self._reset_nav_state()
            elif self.placement_mode == "start":
                self.grid.set_start(row, col)
                self._reset_nav_state()
                self.status_msg = f"Start set: ({row},{col})"
            elif self.placement_mode == "goal":
                self.grid.set_goal(row, col)
                self._reset_nav_state()
                self.status_msg = f"Goal set: ({row},{col})"

        elif event.button == 3:  # Right click — always toggle obstacle
            self.grid.toggle_obstacle(row, col)
            self._reset_nav_state()

    # ──────────────────────────────────────────────────────
    # Navigation Control
    # ──────────────────────────────────────────────────────

    def _start_navigation(self):
        """Initialize and start the navigation."""
        # Reset old nav state
        self._reset_nav_state()
        self.navigator = Navigator(algorithm=self.algorithm, sensor_range=4)

        success = self.navigator.setup(self.grid)
        if success:
            self.nav_ready = True
            self.nav_running = True
            self.nav_done = False
            self.status_msg = f"▶ Navigating with {self.algorithm.upper()}..."
        else:
            self.status_msg = "❌ No path found — try repositioning obstacles"

    def _reset_nav_state(self):
        """Reset navigation without resetting grid obstacles."""
        self.nav_running = False
        self.nav_done = False
        self.nav_ready = False
        self.step_result = {}
        if self.grid.start and self.grid.goal:
            self.grid.clear_path_marks()
        self.status_msg = "Ready — Press SPACE to plan & navigate"

    def _reset_random(self):
        """Generate new random obstacles and reset navigation."""
        seed = random.randint(0, 9999)
        self.grid.generate_random_obstacles(density=self.obstacle_density, seed=seed)
        self.grid.set_start(1, 1)
        self.grid.set_goal(self.grid_size - 2, self.grid_size - 2)
        self._reset_nav_state()
        self.status_msg = "🔀 New random map — Press SPACE"

    def _clear_obstacles(self):
        """Remove all obstacles."""
        start_bk = self.grid.start
        goal_bk = self.grid.goal
        self.grid.reset()
        if start_bk:
            self.grid.set_start(*start_bk)
        if goal_bk:
            self.grid.set_goal(*goal_bk)
        self._reset_nav_state()
        self.status_msg = "Cleared all obstacles"

    def _toggle_algorithm(self):
        """Switch between A* and Dijkstra."""
        self.algo_idx = (self.algo_idx + 1) % len(self.algo_names)
        self.algorithm = self.algo_names[self.algo_idx]
        self._reset_nav_state()
        self.status_msg = f"Algorithm: {self.algorithm.upper()} (press SPACE)"

    # ──────────────────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────────────────

    def _draw(self):
        """Render the complete frame."""
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_sensor_range()
        self._draw_hud()

    def _draw_grid(self):
        """Draw all grid cells."""
        cs = self.cell_size

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_type = self.grid.cells[r][c]
                color = CELL_TYPE_COLORS.get(cell_type, FREE_COLOR)

                rect = pygame.Rect(c * cs + 1, r * cs + 1, cs - 2, cs - 2)

                # Draw cell with rounded corners effect
                pygame.draw.rect(self.screen, color, rect, border_radius=4)

                # Add subtle icons for special cells
                cx = c * cs + cs // 2
                cy = r * cs + cs // 2

                if cell_type == AGENT:
                    # Draw agent as a circle
                    pygame.draw.circle(self.screen, AGENT_COLOR, (cx, cy), cs // 3)
                    pygame.draw.circle(self.screen, (255, 200, 100), (cx, cy), cs // 5)

                elif cell_type == START:
                    self._draw_text_centered("S", cx, cy, START_COLOR,
                                             bold=True, size="large")

                elif cell_type == GOAL:
                    self._draw_text_centered("G", cx, cy, GOAL_COLOR,
                                             bold=True, size="large")

                elif cell_type == PATH:
                    # Dot in path cells
                    pygame.draw.circle(self.screen, PATH_COLOR, (cx, cy), 3)

        # Grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                self.screen, GRID_LINE,
                (0, i * cs), (self.grid_px, i * cs)
            )
            pygame.draw.line(
                self.screen, GRID_LINE,
                (i * cs, 0), (i * cs, self.grid_px)
            )

    def _draw_sensor_range(self):
        """Draw transparent sensor range circle around agent."""
        if not self.nav_running and not self.nav_done:
            return
        if not self.navigator.agent.position:
            return

        r, c = self.navigator.agent.position
        cx = c * self.cell_size + self.cell_size // 2
        cy = r * self.cell_size + self.cell_size // 2
        radius = self.navigator.detector.sensor_range * self.cell_size

        # Transparent circle using a surface
        surf = pygame.Surface((self.grid_px, self.grid_px), pygame.SRCALPHA)
        pygame.draw.circle(surf, (100, 149, 237, 25), (cx, cy), radius)
        pygame.draw.circle(surf, (100, 149, 237, 80), (cx, cy), radius, 1)
        self.screen.blit(surf, (0, 0))

        # Danger ring if in danger zone
        if self.navigator.detector.danger_zone:
            pygame.draw.circle(
                self.screen, (255, 50, 50),
                (cx, cy),
                2 * self.cell_size, 2
            )

    def _draw_hud(self):
        """Draw the information panel on the right side."""
        hud_x = self.grid_px + 10
        y = 15
        line_h = 20

        def text(txt, color=(200, 200, 255), bold=False, small=False):
            nonlocal y
            font = self.font_small if small else (
                self.font_title if bold else self.font_body
            )
            surf = font.render(txt, True, color)
            self.screen.blit(surf, (hud_x, y))
            y += line_h

        def sep():
            nonlocal y
            pygame.draw.line(
                self.screen, (50, 50, 90),
                (hud_x, y), (hud_x + HUD_WIDTH - 20, y)
            )
            y += 8

        # Title
        text("AI NAVIGATION SIM", (100, 200, 255), bold=True)
        sep()

        # Algorithm
        algo_color = (100, 255, 100) if self.algorithm == "astar" else (255, 200, 0)
        text(f"Algorithm : {self.algorithm.upper()}", algo_color)
        text(f"Grid Size : {self.grid_size}x{self.grid_size}")
        text(f"Obstacles : {int(self.obstacle_density*100)}%")
        sep()

        # Status
        status_color = (100, 255, 100) if "Goal" in self.status_msg else (255, 220, 100)
        text("STATUS:", (180, 180, 255), bold=True)
        # Wrap long status
        words = self.status_msg
        text(f"  {words[:35]}", status_color)
        if len(words) > 35:
            text(f"  {words[35:70]}", status_color)
        sep()

        # Agent stats
        if self.step_result:
            stats = self.step_result.get("agent_stats", {})
            text("AGENT STATS:", (180, 180, 255), bold=True)
            text(f"  Position   : {stats.get('position', '-')}")
            text(f"  Steps Taken: {stats.get('steps', 0)}")
            text(f"  Distance   : {stats.get('total_distance', 0):.1f}")
            text(f"  Replans    : {self.step_result.get('replan_count', 0)}")
            text(f"  Heading    : {stats.get('heading', '-')}")
            text(f"  Status     : {stats.get('status', '-')}")
            sep()

            scan = self.step_result.get("scan", {})
            text("SENSOR:", (180, 180, 255), bold=True)
            n_obs = len(scan.get("detected_obstacles", []))
            danger = "🚨 YES" if scan.get("danger_zone") else "No"
            text(f"  Obstacles  : {n_obs} detected")
            text(f"  Nearest    : {scan.get('nearest_distance', '-')}")
            text(f"  Danger Zone: {danger}")
            sep()

        # Planner stats
        if self.nav_ready:
            explored = getattr(self.navigator.planner, "explored_count", 0)
            path_len = len(self.navigator.current_path)
            text("PATH PLANNER:", (180, 180, 255), bold=True)
            text(f"  Path Length  : {path_len} cells")
            text(f"  Cells Explored: {explored}")
            sep()

        # Controls
        text("CONTROLS:", (180, 180, 255), bold=True)
        controls = [
            ("SPACE", "Plan & Navigate"),
            ("Click", "Toggle Obstacle"),
            ("R", "Random Reset"),
            ("C", "Clear Map"),
            ("TAB", "Switch Algorithm"),
            ("F2/F3", "Set Start/Goal"),
            ("S", "Save Screenshot"),
            ("Q/ESC", "Quit"),
        ]
        for key, desc in controls:
            text(f"  [{key}] {desc}", (150, 150, 200), small=True)

        # Legend
        sep()
        text("LEGEND:", (180, 180, 255), bold=True)
        legend = [
            (START_COLOR,    "Start"),
            (GOAL_COLOR,     "Goal"),
            (AGENT_COLOR,    "Agent"),
            (PATH_COLOR,     "Path"),
            (OBSTACLE_COLOR, "Obstacle"),
            (VISITED_COLOR,  "Explored"),
        ]
        for color, label in legend:
            pygame.draw.rect(
                self.screen, color,
                pygame.Rect(hud_x, y + 3, 12, 12), border_radius=2
            )
            lbl = self.font_small.render(label, True, (180, 180, 200))
            self.screen.blit(lbl, (hud_x + 18, y))
            y += 18

    def _draw_text_centered(
        self, text: str, cx: int, cy: int,
        color: tuple, bold: bool = False, size: str = "normal"
    ):
        """Render text centered at pixel coordinates."""
        font = self.font_large if size == "large" else (
            self.font_title if bold else self.font_body
        )
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(cx, cy))
        self.screen.blit(surf, rect)

    # ──────────────────────────────────────────────────────
    # Output
    # ──────────────────────────────────────────────────────

    def _save_screenshot(self):
        """Save current frame to outputs/ folder."""
        os.makedirs("outputs", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("outputs", f"nav_screenshot_{timestamp}.png")
        pygame.image.save(self.screen, path)
        self.status_msg = f"📸 Saved: {path}"
        logger.info(f"Screenshot saved: {path}")

    def _quit(self):
        """Clean exit."""
        logger.info("Simulation ended.")
        # Print final summary if navigation ran
        if self.nav_ready:
            summary = self.navigator.get_summary()
            logger.info(f"Final Summary: {summary}")
        pygame.quit()
        sys.exit(0)
