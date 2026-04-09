# ============================================================
# notebooks/02_matplotlib_visualization.py
# ============================================================
# PURPOSE:
#   Generates high-quality matplotlib charts showing:
#   - Grid environment with planned path
#   - A* vs Dijkstra comparison plot
#   - Navigation metrics bar chart
#   - Sensor scan heatmap
#
# Run: python notebooks/02_matplotlib_visualization.py
# Saves to: outputs/visualization_*.png
# ============================================================

import sys
import os

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")        # Save to file (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation.grid import Grid, FREE, OBSTACLE, START, GOAL, PATH, VISITED
from src.path_planning.astar import AStarPlanner
from src.path_planning.dijkstra import DijkstraPlanner
from src.perception.obstacle_detector import ObstacleDetector
from src.navigation.navigator import Navigator

os.makedirs("outputs", exist_ok=True)

# ── Color setup ────────────────────────────────────────────
CELL_ORDER = [FREE, OBSTACLE, START, GOAL, PATH, VISITED]
COLORS = ["#1e1e2e", "#f38ba8", "#a6e3a1", "#fab387", "#89b4fa", "#313244"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(list(range(len(CELL_ORDER) + 1)), CMAP.N)

DARK_BG = "#1e1e2e"
TEXT_COLOR = "#cdd6f4"
ACCENT = "#89b4fa"


def grid_to_image(grid):
    """Convert grid cells to indexed array for imshow."""
    arr = np.zeros((grid.rows, grid.cols))
    for r in range(grid.rows):
        for c in range(grid.cols):
            v = grid.cells[r][c]
            arr[r][c] = CELL_ORDER.index(v) if v in CELL_ORDER else 0
    return arr


def setup_ax(ax, title=""):
    """Apply dark theme to axes."""
    ax.set_facecolor("#181825")
    ax.tick_params(colors="#6c7086")
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=6)


# ── Plot 1: Path Planning Grid ─────────────────────────────

def plot_path_grid():
    print("📊 Generating path planning visualization...")

    g = Grid(20, 20)
    g.generate_random_obstacles(density=0.22, seed=42)
    g.set_start(1, 1)
    g.set_goal(18, 18)

    planner = AStarPlanner(allow_diagonal=False)
    path = planner.find_path(g)

    # Mark visited cells
    for cell in planner.visited_cells:
        r, c = cell
        if g.cells[r][c] == FREE:
            g.cells[r][c] = VISITED

    if path:
        g.mark_path(path)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=DARK_BG)
    setup_ax(ax, "A* Path Planning — 20×20 Grid")

    img = grid_to_image(g)
    ax.imshow(img, cmap=CMAP, norm=NORM, interpolation="nearest")

    # Annotate start/goal
    if g.start:
        ax.text(g.start[1], g.start[0], "S", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    if g.goal:
        ax.text(g.goal[1], g.goal[0], "G", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])

    handles = [
        mpatches.Patch(color="#1e1e2e", label="Free"),
        mpatches.Patch(color="#f38ba8", label="Obstacle"),
        mpatches.Patch(color="#a6e3a1", label="Start (S)"),
        mpatches.Patch(color="#fab387", label="Goal (G)"),
        mpatches.Patch(color="#89b4fa", label="Path"),
        mpatches.Patch(color="#313244", label="Explored"),
    ]
    ax.legend(handles=handles, loc="upper right",
              facecolor="#313244", labelcolor=TEXT_COLOR,
              fontsize=8, framealpha=0.9)

    info = (f"Path: {len(path) if path else 'NONE'} steps  |  "
            f"Explored: {planner.explored_count} cells")
    ax.set_xlabel(info, color="#6c7086", fontsize=9)

    plt.tight_layout()
    out = "outputs/visualization_path_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"   ✅ Saved: {out}")
    plt.close()


# ── Plot 2: A* vs Dijkstra Comparison ─────────────────────

def plot_algorithm_comparison():
    print("📊 Generating A* vs Dijkstra comparison...")

    sizes = [5, 10, 15, 20, 25, 30]
    astar_explored = []
    dijkstra_explored = []
    astar_lengths = []
    dijkstra_lengths = []

    for n in sizes:
        g = Grid(n, n)
        g.generate_random_obstacles(density=0.2, seed=42)
        g.set_start(0, 0)
        g.set_goal(n - 1, n - 1)

        g1 = copy.deepcopy(g)
        g2 = copy.deepcopy(g)

        ap = AStarPlanner()
        dp = DijkstraPlanner()
        pa = ap.find_path(g1)
        pd = dp.find_path(g2)

        astar_explored.append(ap.explored_count)
        dijkstra_explored.append(dp.explored_count)
        astar_lengths.append(len(pa) if pa else 0)
        dijkstra_lengths.append(len(pd) if pd else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)

    # Chart 1: Cells explored
    setup_ax(ax1, "Cells Explored — A* vs Dijkstra")
    ax1.plot(sizes, astar_explored, "o-", color="#89b4fa", label="A*", linewidth=2)
    ax1.plot(sizes, dijkstra_explored, "s--", color="#f38ba8", label="Dijkstra", linewidth=2)
    ax1.set_xlabel("Grid Size (N×N)", color="#6c7086")
    ax1.set_ylabel("Cells Explored", color="#6c7086")
    ax1.legend(facecolor="#313244", labelcolor=TEXT_COLOR)
    ax1.fill_between(sizes, astar_explored, dijkstra_explored,
                     alpha=0.15, color="#89b4fa")

    # Chart 2: Path lengths
    setup_ax(ax2, "Path Length — A* vs Dijkstra")
    ax2.plot(sizes, astar_lengths, "o-", color="#a6e3a1", label="A*", linewidth=2)
    ax2.plot(sizes, dijkstra_lengths, "s--", color="#fab387", label="Dijkstra", linewidth=2)
    ax2.set_xlabel("Grid Size (N×N)", color="#6c7086")
    ax2.set_ylabel("Path Length (steps)", color="#6c7086")
    ax2.legend(facecolor="#313244", labelcolor=TEXT_COLOR)

    fig.suptitle("Algorithm Efficiency Comparison", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = "outputs/visualization_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"   ✅ Saved: {out}")
    plt.close()


# ── Plot 3: Sensor Scan Heatmap ────────────────────────────

def plot_sensor_heatmap():
    print("📊 Generating sensor scan heatmap...")

    g = Grid(15, 15)
    g.generate_random_obstacles(density=0.25, seed=17)
    g.set_start(7, 7)

    det = ObstacleDetector(sensor_range=5)
    result = det.scan(g, (7, 7))

    # Build heatmap: 0=free, 0.5=scanned, 1=obstacle
    heatmap = np.zeros((15, 15))
    for r in range(15):
        for c in range(15):
            if g.cells[r][c] == OBSTACLE:
                heatmap[r][c] = 1.0
            elif (r, c) in [(r, c) for r_ in range(15) for c in range(15)
                            if (r_-7)**2 + (c-7)**2 <= 25]:
                heatmap[r][c] = 0.3

    for r_, c_ in result["detected_obstacles"]:
        heatmap[r_][c_] = 0.9
    heatmap[7][7] = 0.6  # Agent

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=DARK_BG)
    setup_ax(ax, "Sensor Scan Heatmap (range=5)")

    im = ax.imshow(heatmap, cmap="plasma", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Obstacle Intensity", color=TEXT_COLOR)

    # Agent marker
    ax.scatter(7, 7, c="#a6e3a1", s=200, marker="*", zorder=5, label="Agent")
    # Obstacle markers
    for r_, c_ in result["detected_obstacles"]:
        ax.scatter(c_, r_, c="#f38ba8", s=50, marker="x", zorder=4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        f"Detected: {len(result['detected_obstacles'])} obstacles | "
        f"Danger: {'YES 🚨' if result['danger_zone'] else 'No'}",
        color="#6c7086", fontsize=9
    )

    plt.tight_layout()
    out = "outputs/visualization_sensor.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"   ✅ Saved: {out}")
    plt.close()


# ── Plot 4: Navigation Journey Chart ──────────────────────

def plot_navigation_journey():
    print("📊 Generating navigation journey chart...")

    g = Grid(15, 15)
    g.generate_random_obstacles(density=0.15, seed=3)
    g.set_start(0, 0)
    g.set_goal(14, 14)

    nav = Navigator(algorithm="astar", sensor_range=4)
    success = nav.setup(g)

    if not success:
        print("   ⚠️  No path found — skipping journey chart")
        return

    history = []
    scan_sizes = []
    steps = 0
    while steps < 500:
        result = nav.step(g)
        history.append(result["agent_stats"]["position"])
        scan_sizes.append(len(result["scan"].get("detected_obstacles", [])))
        steps += 1
        if result["status"] in ("GOAL_REACHED", "STUCK"):
            break

    fig = plt.figure(figsize=(12, 5), facecolor=DARK_BG)
    gs = GridSpec(1, 2, figure=fig)

    # Chart 1: Agent path on grid
    ax1 = fig.add_subplot(gs[0])
    setup_ax(ax1, "Agent Navigation Path")
    img = grid_to_image(g)
    ax1.imshow(img, cmap=CMAP, norm=NORM, interpolation="nearest")
    if history:
        ys, xs = zip(*history)
        ax1.plot(xs, ys, "w-", linewidth=1, alpha=0.5)
        ax1.scatter(xs[0], ys[0], c="#a6e3a1", s=80, zorder=5, label="Start")
        ax1.scatter(xs[-1], ys[-1], c="#fab387", s=80, marker="*", zorder=5, label="End")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(facecolor="#313244", labelcolor=TEXT_COLOR, fontsize=8)

    # Chart 2: Obstacles detected per step
    ax2 = fig.add_subplot(gs[1])
    setup_ax(ax2, "Obstacles Detected per Step")
    ax2.fill_between(range(len(scan_sizes)), scan_sizes, color="#f38ba8", alpha=0.7)
    ax2.plot(range(len(scan_sizes)), scan_sizes, color="#f38ba8", linewidth=1)
    ax2.axhline(y=2, color="#fab387", linestyle="--", alpha=0.5, label="Danger threshold")
    ax2.set_xlabel("Navigation Step", color="#6c7086")
    ax2.set_ylabel("Detected Obstacles", color="#6c7086")
    ax2.legend(facecolor="#313244", labelcolor=TEXT_COLOR, fontsize=8)

    fig.suptitle("Navigation Journey Analysis", color=TEXT_COLOR,
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = "outputs/visualization_journey.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"   ✅ Saved: {out}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AI Navigation — Matplotlib Visualization Suite")
    print("=" * 55)

    plot_path_grid()
    plot_algorithm_comparison()
    plot_sensor_heatmap()
    plot_navigation_journey()

    print("\n" + "=" * 55)
    print("  All visualizations saved to outputs/ folder!")
    print("=" * 55)
