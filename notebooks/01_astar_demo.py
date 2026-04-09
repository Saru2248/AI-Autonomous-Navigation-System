# ============================================================
# notebooks/01_astar_demo.py
# ============================================================
# PURPOSE:
#   Pure-Python demo script showing A* path planning step
#   by step with ASCII visualization in the terminal.
#   No dependencies except standard library + numpy.
#
# Run: python notebooks/01_astar_demo.py
# ============================================================

import sys
import os

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Make sure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation.grid import Grid, FREE, OBSTACLE, START, GOAL, PATH
from src.path_planning.astar import AStarPlanner


def print_grid(grid: Grid, title: str = ""):
    """ASCII art visualization of the grid."""
    symbols = {
        FREE:     "·",
        OBSTACLE: "█",
        START:    "S",
        GOAL:     "G",
        PATH:     "○",
    }
    print(f"\n{'='*40}")
    if title:
        print(f"  {title}")
        print(f"{'='*40}")
    for r in range(grid.rows):
        row = " ".join(symbols.get(grid.cells[r][c], "?") for c in range(grid.cols))
        print(f"  {row}")
    print(f"{'='*40}")


def demo_basic():
    """Demo 1: Simple 10x10 grid, no obstacles."""
    print("\n🔹 DEMO 1: Basic A* on empty 10×10 grid")
    g = Grid(10, 10)
    g.set_start(0, 0)
    g.set_goal(9, 9)

    planner = AStarPlanner(allow_diagonal=False)
    path = planner.find_path(g)

    g.mark_path(path)
    print_grid(g, "A* Path (no obstacles)")
    print(f"  Path length   : {len(path)} steps")
    print(f"  Cells explored: {planner.explored_count}")


def demo_with_obstacles():
    """Demo 2: 15x15 grid with a wall of obstacles."""
    print("\n🔹 DEMO 2: A* with horizontal wall (gap on right)")
    g = Grid(15, 15)
    g.set_start(0, 0)
    g.set_goal(14, 14)

    # Horizontal wall at row 7, gap at (7, 14)
    for c in range(13):
        g.place_obstacle(7, c)

    planner = AStarPlanner(allow_diagonal=False)
    path = planner.find_path(g)

    if path:
        g.mark_path(path)
        print_grid(g, "A* Path (with wall)")
        print(f"  Path length   : {len(path)} steps")
        print(f"  Cells explored: {planner.explored_count}")
    else:
        print("  ❌ No path found!")


def demo_random():
    """Demo 3: Random obstacle grid."""
    print("\n🔹 DEMO 3: Random obstacles (seed=42)")
    g = Grid(12, 12)
    g.set_start(0, 0)
    g.set_goal(11, 11)
    g.generate_random_obstacles(density=0.2, seed=42)

    planner = AStarPlanner(allow_diagonal=False)
    path = planner.find_path(g)

    if path:
        g.mark_path(path)
        print_grid(g, "A* Random Grid")
        print(f"  Path length   : {len(path)} steps")
        print(f"  Cells explored: {planner.explored_count}")
    else:
        print("  ❌ No path found (too many obstacles!)")


def demo_comparison():
    """Demo 4: Compare A* vs Dijkstra on same grid."""
    from src.path_planning.dijkstra import DijkstraPlanner
    import copy

    print("\n🔹 DEMO 4: A* vs Dijkstra Comparison")
    g = Grid(15, 15)
    g.generate_random_obstacles(density=0.25, seed=7)
    g.set_start(0, 0)
    g.set_goal(14, 14)

    g_astar = copy.deepcopy(g)
    g_dijkstra = copy.deepcopy(g)

    astar = AStarPlanner(allow_diagonal=False)
    dijkstra = DijkstraPlanner(allow_diagonal=False)

    path_a = astar.find_path(g_astar)
    path_d = dijkstra.find_path(g_dijkstra)

    print(f"\n  {'Algorithm':<15} {'Path Length':<15} {'Cells Explored':<15}")
    print(f"  {'-'*45}")

    if path_a:
        print(f"  {'A*':<15} {len(path_a):<15} {astar.explored_count:<15}")
    else:
        print(f"  {'A*':<15} {'NO PATH':<15} {astar.explored_count:<15}")

    if path_d:
        print(f"  {'Dijkstra':<15} {len(path_d):<15} {dijkstra.explored_count:<15}")
    else:
        print(f"  {'Dijkstra':<15} {'NO PATH':<15} {dijkstra.explored_count:<15}")

    if path_a and path_d:
        efficiency = dijkstra.explored_count / max(astar.explored_count, 1)
        print(f"\n  ✅ A* explored {efficiency:.1f}x FEWER cells than Dijkstra!")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AI Autonomous Navigation — A* Planner Demos")
    print("=" * 55)

    demo_basic()
    demo_with_obstacles()
    demo_random()
    demo_comparison()

    print("\n✅ All demos complete!")
