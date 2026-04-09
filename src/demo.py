# ============================================================
# src/demo.py
# ============================================================
# PURPOSE:
#   Headless demo mode — runs the full navigation pipeline
#   without user interaction and saves results as PNG files.
#   Perfect for GitHub CI, screenshots, and quick proof.
#
# Run via:  python main.py --mode demo --save-output
# ============================================================

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation.grid import (
    Grid, FREE, OBSTACLE, START, GOAL, PATH, VISITED, AGENT
)
from src.navigation.navigator import Navigator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Color map for grid cell types ──────────────────────────
CELL_COLORS = {
    FREE:     "#1e1e2e",    # Dark background
    OBSTACLE: "#f38ba8",    # Red
    START:    "#a6e3a1",    # Green
    GOAL:     "#fab387",    # Orange
    PATH:     "#89b4fa",    # Blue
    VISITED:  "#313244",    # Dark gray
    AGENT:    "#cba6f7",    # Purple
}

# Map cell int → 0..6 index, define ordered colormap
CELL_ORDER = [FREE, OBSTACLE, START, GOAL, PATH, VISITED, AGENT]
COLOR_LIST = [CELL_COLORS[c] for c in CELL_ORDER]
CMAP = ListedColormap(COLOR_LIST)


def _grid_to_array(grid: Grid) -> np.ndarray:
    """Convert grid cells to a 0-indexed numpy array for imshow."""
    arr = np.zeros((grid.rows, grid.cols), dtype=int)
    for r in range(grid.rows):
        for c in range(grid.cols):
            val = grid.cells[r][c]
            arr[r][c] = CELL_ORDER.index(val) if val in CELL_ORDER else 0
    return arr


def _render_grid(grid: Grid, title: str, ax: plt.Axes):
    """Render grid as a colored heatmap."""
    arr = _grid_to_array(grid)
    ax.imshow(arr, cmap=CMAP, vmin=0, vmax=len(CELL_ORDER) - 1, interpolation="nearest")
    ax.set_title(title, color="#cdd6f4", fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")

    # Overlay start/goal labels
    if grid.start:
        r, c = grid.start
        ax.text(c, r, "S", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    if grid.goal:
        r, c = grid.goal
        ax.text(c, r, "G", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")


def run_demo_mode(
    grid_size: int = 20,
    obstacle_density: float = 0.25,
    save_output: bool = True
):
    """
    Run a complete headless navigation demo.

    Steps:
      1. Build grid with random obstacles
      2. Run A* path planning
      3. Simulate agent navigation step by step
      4. Capture before/after snapshots
      5. Save composite figure to outputs/

    Args:
        grid_size: NxN grid dimensions.
        obstacle_density: Fraction of cells that are obstacles.
        save_output: If True, save PNG files to outputs/.
    """
    logger.info("=== DEMO MODE STARTED ===")
    os.makedirs("outputs", exist_ok=True)

    seed = random.randint(0, 99999)
    logger.info(f"Grid: {grid_size}x{grid_size}, density={obstacle_density}, seed={seed}")

    # ── 1. Build Grid ──────────────────────────────────────
    grid = Grid(grid_size, grid_size)
    grid.generate_random_obstacles(density=obstacle_density, seed=seed)
    grid.set_start(1, 1)
    grid.set_goal(grid_size - 2, grid_size - 2)

    # snapshot: initial map
    import copy
    grid_initial = copy.deepcopy(grid)

    # ── 2. Path Planning ───────────────────────────────────
    navigator = Navigator(algorithm="astar", sensor_range=4)
    success = navigator.setup(grid)

    if not success:
        logger.error("No path found! Adjust obstacle density.")
        _save_no_path_figure(grid_initial, save_output)
        return

    logger.info(f"Path found: {len(navigator.current_path)} steps")

    # snapshot: path planned
    import copy
    grid_planned = copy.deepcopy(grid)

    # ── 3. Navigation Simulation ───────────────────────────
    max_steps = grid_size * grid_size * 2
    step_count = 0
    final_status = "UNKNOWN"
    stats_list = []

    while step_count < max_steps:
        result = navigator.step(grid)
        final_status = result["status"]
        stats_list.append(result["agent_stats"])
        step_count += 1
        if final_status in ("GOAL_REACHED", "STUCK"):
            break

    logger.info(f"Navigation complete: {final_status} in {step_count} steps")

    # snapshot: final state
    import copy
    grid_final = copy.deepcopy(grid)

    # ── 4. Build & Save Figure ─────────────────────────────
    fig = plt.figure(figsize=(15, 10), facecolor="#1e1e2e")

    # Row 1: grid states
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)

    _render_grid(grid_initial, "① Initial Environment", ax1)
    _render_grid(grid_planned, "② Path Planned (A*)", ax2)
    _render_grid(grid_final,   "③ Final Navigation Result", ax3)

    # Row 2: stats chart
    ax4 = fig.add_subplot(2, 3, 4)
    _plot_steps_chart(stats_list, ax4)

    ax5 = fig.add_subplot(2, 3, 5)
    _plot_summary_bar(navigator, step_count, final_status, ax5)

    ax6 = fig.add_subplot(2, 3, 6)
    _plot_scan_info(navigator, ax6)

    # Title
    status_symbol = "✅" if final_status == "GOAL_REACHED" else "❌"
    fig.suptitle(
        f"AI Autonomous Navigation System — Demo  |  {status_symbol} {final_status}",
        color="#cdd6f4", fontsize=14, fontweight="bold", y=0.98
    )

    # Legend
    handles = [
        mpatches.Patch(color=CELL_COLORS[FREE],     label="Free"),
        mpatches.Patch(color=CELL_COLORS[OBSTACLE], label="Obstacle"),
        mpatches.Patch(color=CELL_COLORS[START],    label="Start"),
        mpatches.Patch(color=CELL_COLORS[GOAL],     label="Goal"),
        mpatches.Patch(color=CELL_COLORS[PATH],     label="Planned Path"),
        mpatches.Patch(color=CELL_COLORS[VISITED],  label="Explored"),
        mpatches.Patch(color=CELL_COLORS[AGENT],    label="Agent"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               facecolor="#313244", labelcolor="#cdd6f4",
               framealpha=0.9, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_output:
        out_path = os.path.join("outputs", "demo_result.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info(f"Demo result saved: {out_path}")
        print(f"\n✅ Demo complete! Output saved to: {out_path}")
    else:
        plt.show()

    plt.close(fig)

    # Print summary to terminal
    _print_terminal_summary(navigator, step_count, final_status)


# ── Plotting Helpers ───────────────────────────────────────

def _plot_steps_chart(stats_list, ax):
    """Bar chart showing distance per step."""
    ax.set_facecolor("#181825")
    if not stats_list:
        ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes, color="gray")
        return

    steps = [s.get("steps", i) for i, s in enumerate(stats_list)]
    distances = [s.get("total_distance", 0) for s in stats_list]

    ax.plot(steps, distances, color="#89b4fa", linewidth=2)
    ax.fill_between(steps, distances, alpha=0.3, color="#89b4fa")
    ax.set_title("Distance Traveled per Step", color="#cdd6f4", fontsize=9)
    ax.tick_params(colors="#6c7086")
    ax.set_xlabel("Step", color="#6c7086", fontsize=8)
    ax.set_ylabel("Total Distance", color="#6c7086", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")


def _plot_summary_bar(navigator, step_count, final_status, ax):
    """Horizontal bar chart with key metrics."""
    ax.set_facecolor("#181825")
    summary = navigator.get_summary()
    metrics = {
        "Steps Taken": step_count,
        "Path Length": summary.get("path_length_planned", 0),
        "Cells Explored": summary.get("cells_explored", 0),
        "Replans": summary.get("replan_count", 0),
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#89b4fa", "#a6e3a1", "#fab387", "#f38ba8"]
    bars = ax.barh(labels, values, color=colors, height=0.5)
    ax.set_title("Navigation Metrics", color="#cdd6f4", fontsize=9)
    ax.tick_params(colors="#6c7086")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color="#cdd6f4", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")


def _plot_scan_info(navigator, ax):
    """Pie chart of algorithm efficiency."""
    ax.set_facecolor("#181825")
    summary = navigator.get_summary()
    explored = summary.get("cells_explored", 1)
    path_len = summary.get("path_length_planned", 1)
    not_on_path = max(0, explored - path_len)

    sizes = [path_len, not_on_path]
    labels = ["On Path", "Explored (Off Path)"]
    colors = ["#89b4fa", "#313244"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "#cdd6f4", "fontsize": 8}
    )
    for at in autotexts:
        at.set_color("#cdd6f4")
    ax.set_title("A* Exploration Efficiency", color="#cdd6f4", fontsize=9)


def _save_no_path_figure(grid_initial, save_output):
    """Save error figure when no path is found."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="#1e1e2e")
    _render_grid(grid_initial, "❌ No Path Found — Too Many Obstacles!", ax)
    plt.tight_layout()
    if save_output:
        path = os.path.join("outputs", "demo_no_path.png")
        plt.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info(f"No-path figure saved: {path}")
    plt.close(fig)


def _print_terminal_summary(navigator, step_count, final_status):
    """Pretty-print summary to terminal."""
    summary = navigator.get_summary()
    print("\n" + "=" * 55)
    print("  AI AUTONOMOUS NAVIGATION — DEMO SUMMARY")
    print("=" * 55)
    print(f"  Status         : {final_status}")
    print(f"  Algorithm      : {summary.get('algorithm', '?').upper()}")
    print(f"  Steps Taken    : {step_count}")
    print(f"  Path Length    : {summary.get('path_length_planned', '?')} cells")
    print(f"  Cells Explored : {summary.get('cells_explored', '?')}")
    print(f"  Replans        : {summary.get('replan_count', 0)}")
    print(f"  Total Distance : {summary.get('total_distance', 0):.2f}")
    print("=" * 55)
    print("  Output saved to: outputs/demo_result.png")
    print("=" * 55 + "\n")
