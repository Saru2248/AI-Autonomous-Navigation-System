# ============================================================
# main.py — Entry point for AI Autonomous Navigation System
# ============================================================
# This file ties together all modules:
#   - Simulation environment (pygame grid)
#   - Perception / obstacle detection
#   - Path planning (A* algorithm)
#   - Navigation & control
#   - Visualization
#
# Run: python main.py
# ============================================================

import sys
import os
import argparse

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.simulation.environment import SimulationEnvironment
from src.navigation.navigator import Navigator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Autonomous Navigation System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "demo", "test"],
        default="sim",
        help=(
            "Run mode:\n"
            "  sim   - Full interactive simulation (default)\n"
            "  demo  - Auto-run demo with random obstacles\n"
            "  test  - Run unit tests for all modules"
        )
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=20,
        help="Grid size (NxN). Default: 20"
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=35,
        help="Pixel size of each grid cell. Default: 35"
    )
    parser.add_argument(
        "--obstacle-density",
        type=float,
        default=0.2,
        help="Fraction of cells that are obstacles (0.0-0.5). Default: 0.2"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Simulation frames per second. Default: 10"
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save final grid screenshot to outputs/ folder"
    )
    return parser.parse_args()


def run_simulation(args):
    """Launch the full interactive pygame simulation."""
    logger.info("Starting AI Autonomous Navigation Simulation...")
    logger.info(f"Grid: {args.grid_size}x{args.grid_size}, "
                f"Cell Size: {args.cell_size}px, "
                f"Obstacle Density: {args.obstacle_density}")

    env = SimulationEnvironment(
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        obstacle_density=args.obstacle_density,
        fps=args.fps,
        save_output=args.save_output
    )
    env.run()


def run_demo(args):
    """Auto-demo mode — no user interaction needed."""
    logger.info("Running DEMO mode...")
    from src.demo import run_demo_mode
    run_demo_mode(
        grid_size=args.grid_size,
        obstacle_density=args.obstacle_density,
        save_output=args.save_output
    )


def run_tests():
    """Run all module unit tests."""
    logger.info("Running module tests...")
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=False
    )
    sys.exit(result.returncode)


def main():
    args = parse_args()

    print("=" * 60)
    print("  AI-Based Autonomous Navigation System")
    print("  Mode:", args.mode.upper())
    print("=" * 60)

    if args.mode == "sim":
        run_simulation(args)
    elif args.mode == "demo":
        run_demo(args)
    elif args.mode == "test":
        run_tests()


if __name__ == "__main__":
    main()
