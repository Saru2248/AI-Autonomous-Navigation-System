import pytest
from src.simulation.grid import Grid
from src.path_planning.q_learning import QLearningPlanner

def test_qlearning_basic_path():
    grid = Grid(5, 5)
    planner = QLearningPlanner(episodes=500, epsilon=0.5)
    path = planner.find_path(grid, start=(0, 0), goal=(4, 4))
    
    # Q-Learning should eventually find a path in a small grid
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)

def test_qlearning_no_path():
    grid = Grid(5, 5)
    # Block path
    for c in range(5):
        grid.toggle_obstacle(2, c)
        
    planner = QLearningPlanner(episodes=100)
    path = planner.find_path(grid, start=(0, 0), goal=(4, 4))
    
    assert path is None
