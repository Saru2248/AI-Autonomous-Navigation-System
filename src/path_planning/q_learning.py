# ============================================================
# src/path_planning/q_learning.py
# ============================================================
# PURPOSE:
#   Implements a Reinforcement Learning (Q-Learning) agent.
#   Unlike A* and Dijkstra which are classical search algorithms,
#   Q-Learning is a true AI/ML algorithm that learns the optimal
#   path through trial and error (episodes).
#
# HOW IT WORKS:
#   1. Initialize a Q-table for state-action pairs.
#   2. Train for N episodes: 
#      - Agent explores the grid, using epsilon-greedy strategy.
#      - Receives reward for reaching the goal, penalty for obstacles.
#      - Updates Q-values using the Bellman equation.
#   3. After training, the agent extracts the optimal path by 
#      greedily following the highest Q-value at each step.
# ============================================================

import random
from typing import List, Tuple, Optional, Dict
from src.simulation.grid import Grid, OBSTACLE
from src.utils.logger import get_logger

logger = get_logger(__name__)

class QLearningPlanner:
    """
    Q-Learning path planner. True AI model.
    """

    def __init__(
        self, 
        allow_diagonal: bool = False,
        alpha: float = 0.5,       # Learning rate
        gamma: float = 0.9,       # Discount factor
        epsilon: float = 0.2,     # Exploration rate
        episodes: int = 2000      # Training iterations
    ):
        self.allow_diagonal = allow_diagonal
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        
        self.q_table: Dict[Tuple[int, int], List[float]] = {}
        self.visited_cells: List[Tuple[int, int]] = []
        self.explored_count: int = 0
        
        # Define actions
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.allow_diagonal:
            self.actions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def _get_q_values(self, state: Tuple[int, int]) -> List[float]:
        """Get or initialize Q-values for a state."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)
        return self.q_table[state]

    def _get_legal_actions(self, grid: Grid, state: Tuple[int, int]) -> List[int]:
        """Return indices of legal actions from current state."""
        r, c = state
        legal = []
        for i, (dr, dc) in enumerate(self.actions):
            nr, nc = r + dr, c + dc
            if grid.is_valid(nr, nc):
                legal.append(i)
        return legal

    def find_path(
        self,
        grid: Grid,
        start: Tuple[int, int] = None,
        goal: Tuple[int, int] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Train the Q-Learning model, then extract the optimal path.
        """
        start = start or grid.start
        goal = goal or grid.goal

        if start is None or goal is None:
            logger.error("Start or goal not set.")
            return None
            
        if start == goal:
            return [start]
            
        # Reset tracking
        self.q_table.clear()
        self.visited_cells = []
        self.explored_count = 0
        
        logger.info(f"Training Q-Learning model for {self.episodes} episodes...")

        # 1. Training Phase
        for episode in range(self.episodes):
            state = start
            steps = 0
            # Limit steps per episode to avoid infinite loops during early exploration
            max_steps = grid.rows * grid.cols 
            
            while state != goal and steps < max_steps:
                self.explored_count += 1
                q_values = self._get_q_values(state)
                legal_actions = self._get_legal_actions(grid, state)
                
                if not legal_actions:
                    break # Stuck
                
                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    action_idx = random.choice(legal_actions)
                else:
                    # Choose legal action with max Q-value
                    max_q = min(q_values) - 1.0 # Ensure any legal action overrides this
                    best_actions = []
                    for act in legal_actions:
                        if q_values[act] > max_q:
                            max_q = q_values[act]
                            best_actions = [act]
                        elif q_values[act] == max_q:
                            best_actions.append(act)
                    action_idx = random.choice(best_actions)

                # Take action
                dr, dc = self.actions[action_idx]
                next_state = (state[0] + dr, state[1] + dc)
                
                # Determine reward
                reward = -1.0 # Default step penalty
                if next_state == goal:
                    reward = 100.0
                elif grid.cells[next_state[0]][next_state[1]] == OBSTACLE:
                    reward = -100.0
                    next_state = state # Don't move into obstacle
                
                # Track visited for visualization
                if next_state not in self.visited_cells:
                    self.visited_cells.append(next_state)
                
                # Update Q-table
                next_q_values = self._get_q_values(next_state)
                best_next_q = max([next_q_values[a] for a in self._get_legal_actions(grid, next_state)]) if next_state != goal else 0.0
                
                self.q_table[state][action_idx] += self.alpha * (
                    reward + self.gamma * best_next_q - self.q_table[state][action_idx]
                )
                
                state = next_state
                steps += 1
                
                if reward == 100.0:
                    break # Goal reached

        # 2. Path Extraction Phase (Exploitation)
        return self._extract_path(grid, start, goal)

    def _extract_path(self, grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Extract the learned best path."""
        path = [start]
        state = start
        max_extract_steps = grid.rows * grid.cols
        steps = 0
        
        while state != goal and steps < max_extract_steps:
            q_values = self._get_q_values(state)
            legal_actions = self._get_legal_actions(grid, state)
            
            if not legal_actions:
                break
                
            # Filter out actions leading to obstacles during extraction
            safe_actions = []
            for a in legal_actions:
                dr, dc = self.actions[a]
                nr, nc = state[0] + dr, state[1] + dc
                if grid.cells[nr][nc] != OBSTACLE:
                    safe_actions.append(a)
                    
            if not safe_actions:
                break

            # Find best safe action
            max_q = float('-inf')
            best_a = safe_actions[0]
            for a in safe_actions:
                if q_values[a] > max_q:
                    max_q = q_values[a]
                    best_a = a
                    
            dr, dc = self.actions[best_a]
            state = (state[0] + dr, state[1] + dc)
            
            if state in path:
                logger.warning("Q-Learning path loops; aborting extraction.")
                break # Loop detected
                
            path.append(state)
            steps += 1
            
        if state == goal:
            logger.info(f"Q-Learning path extracted! Length: {len(path)}")
            return path
            
        logger.warning("Q-Learning model failed to converge to a valid path.")
        return None

    def get_stats(self) -> dict:
        return {
            "cells_explored": self.explored_count,
            "visited_cells": len(self.visited_cells),
            "episodes_trained": self.episodes
        }
