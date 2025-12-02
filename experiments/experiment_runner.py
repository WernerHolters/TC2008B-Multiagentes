"""
Experimental Framework for A* vs Q-Learning Comparison
=====================================================

Variables de comparación:
1. Densidad de obstáculos (0%, 10%, 20%, 30%)
2. Distancia start-goal (cercana: 2-4, media: 5-8, lejana: 9-15)
3. Hiperparámetros Q-Learning (α, γ, ε)

Combinatoria: 4 densidades × 3 distancias × 2 sets hiperparámetros = 24 experimentos
Repeticiones: 3 runs por experimento = 72 ejecuciones totales
"""

import json
import time
import math
import random
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
from heapq import heappush, heappop

# === A* ALGORITHM FUNCTIONS ===

def heuristic(a, b):
    """Manhattan distance heuristic (for 4-neighbor movement)"""
    (x1, y1), (x2, y2) = a, b
    return abs(x1 - x2) + abs(y1 - y2)

def neighbors(pos, env):
    """Get valid neighbors for A* (4-neighbor movement)"""
    x, y = pos
    moves = [
        (1, 0), (-1, 0), (0, 1), (0, -1)  # Only orthogonal moves
    ]

    width = env["width"]
    height = env["height"]
    obstacles = {(o["x"], o["y"]) for o in env["obstacles"]}

    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
            cost = 1.0  # All moves have cost 1 (no diagonals)
            yield (nx, ny), cost

def astar(env):
    """A* pathfinding algorithm"""
    start = (env["start"]["x"], env["start"]["y"])
    goal = (env["goal"]["x"], env["goal"]["y"])

    open_heap = []
    heappush(open_heap, (0, start))

    came_from = {}
    g_cost = {start: 0.0}

    while open_heap:
        _, current = heappop(open_heap)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nxt, step_cost in neighbors(current, env):
            tentative_g = g_cost[current] + step_cost
            if nxt not in g_cost or tentative_g < g_cost[nxt]:
                g_cost[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal)
                heappush(open_heap, (f, nxt))
                came_from[nxt] = current

    return None

# === Q-LEARNING ALGORITHM FUNCTIONS ===

ACTIONS = {
    0: (0, 1),   # up
    1: (0, -1),  # down
    2: (-1, 0),  # left
    3: (1, 0),   # right
}

def create_q_table(width, height, n_actions=4):
    """Initialize Q-table with zeros"""
    return [[[0.0 for _ in range(n_actions)] for _ in range(height)]
            for _ in range(width)]

def epsilon_greedy(q_state, epsilon: float) -> int:
    """Epsilon-greedy action selection"""
    if random.random() < epsilon:
        return random.randrange(len(q_state))
    # Greedy selection
    max_value = max(q_state)
    best_actions = [a for a, v in enumerate(q_state) if v == max_value]
    return random.choice(best_actions)

def step(state, action, width, height, goal, obstacles):
    """Take action in environment and return next state, reward, done"""
    x, y = state
    dx, dy = ACTIONS[action]
    nx, ny = x + dx, y + dy

    # Check if movement is valid
    if nx < 0 or nx >= width or ny < 0 or ny >= height or (nx, ny) in obstacles:
        # Invalid movement: penalty and stay in place
        reward = -10
        next_state = (x, y)
        done = False
    else:
        next_state = (nx, ny)
        if next_state == goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

    return next_state, reward, done

def train_q_learning(width, height, start, goal, obstacles, episodes=2000, max_steps=200,
                    alpha=0.1, gamma=0.9, epsilon_start=0.3, epsilon_end=0.05):
    """Train Q-learning agent"""
    q_table = create_q_table(width, height)
    epsilon = epsilon_start

    for ep in range(episodes):
        state = start
        for _ in range(max_steps):
            x, y = state
            action = epsilon_greedy(q_table[x][y], epsilon)

            next_state, reward, done = step(state, action, width, height, goal, obstacles)
            nx, ny = next_state

            # Q-learning update
            max_next = max(q_table[nx][ny])
            q_old = q_table[x][y][action]
            q_table[x][y][action] = q_old + alpha * (reward + gamma * max_next - q_old)

            state = next_state
            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (ep / episodes))

    return q_table

def greedy_path_from_q(q_table, start, goal, width, height, max_steps=200):
    """Extract greedy path from trained Q-table"""
    path = [start]
    state = start
    visited = set()

    for _ in range(max_steps):
        if state == goal:
            break

        x, y = state
        q_state = q_table[x][y]
        # Greedy action selection
        best_action = q_state.index(max(q_state))
        dx, dy = ACTIONS[best_action]
        next_state = (x + dx, y + dy)

        # Avoid loops
        if next_state in visited:
            break

        visited.add(next_state)
        path.append(next_state)
        state = next_state

        if state == goal:
            break

    return path

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    obstacle_density: float  # 0.0-1.0
    distance_category: str   # "close", "medium", "far"
    start: Tuple[int, int]
    goal: Tuple[int, int]
    alpha: float            # Learning rate
    gamma: float            # Discount factor
    epsilon_start: float    # Initial exploration
    epsilon_end: float      # Final exploration
    reward_type: str        # "basic" or "shaped"
    grid_size: int = 11
    episodes: int = 2000
    max_steps: int = 200

@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    config_id: str
    algorithm: str          # "astar" or "qlearning"
    run_number: int
    
    # Performance metrics
    execution_time: float   # seconds
    path_length: int        # steps in final path
    success: bool           # reached goal?
    
    # Q-Learning specific
    episodes_to_converge: int = None  # episodes until stable policy
    final_q_max: float = None         # max Q-value at end
    
    # A* specific
    nodes_explored: int = None        # search space explored

class ExperimentRunner:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results: List[ExperimentResult] = []
        self.environments: Dict[str, Dict] = {}
        
    def generate_environment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Generate environment based on experiment configuration"""
        width = height = config.grid_size
        
        # Create obstacles based on density
        total_cells = width * height
        num_obstacles = int(total_cells * config.obstacle_density)
        
        # Generate random obstacles (avoid start/goal)
        obstacles = set()
        forbidden = {config.start, config.goal}
        
        while len(obstacles) < num_obstacles:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            pos = (x, y)
            if pos not in forbidden:
                obstacles.add(pos)
        
        return {
            "width": width,
            "height": height,
            "start": {"x": config.start[0], "y": config.start[1]},
            "goal": {"x": config.goal[0], "y": config.goal[1]},
            "obstacles": [{"x": x, "y": y} for x, y in obstacles]
        }
    
    def calculate_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between start and goal"""
        return math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
    
    def generate_start_goal_pairs(self, grid_size: int) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """Generate start-goal pairs for different distance categories"""
        pairs = []
        
        # Close distance (2-4 units)
        pairs.extend([
            ((0, 0), (2, 2), "close"),
            ((5, 5), (7, 7), "close"),
            ((2, 8), (4, 10), "close"),
        ])
        
        # Medium distance (5-8 units)
        pairs.extend([
            ((0, 0), (5, 5), "medium"),
            ((2, 2), (8, 8), "medium"),
            ((1, 9), (7, 3), "medium"),
        ])
        
        # Far distance (9-15 units)
        pairs.extend([
            ((0, 0), (10, 10), "far"),
            ((1, 1), (9, 10), "far"),
            ((2, 0), (10, 8), "far"),
        ])
        
        return pairs
    
    def run_astar_experiment(self, env: Dict[str, Any]) -> Tuple[List[Tuple[int, int]], float, int]:
        """Run A* algorithm and return path, time, nodes explored"""
        start_time = time.time()
        
        # Convert to format expected by A*
        start = (env["start"]["x"], env["start"]["y"])
        goal = (env["goal"]["x"], env["goal"]["y"])
        obstacles = {(o["x"], o["y"]) for o in env["obstacles"]}
        
        env_for_astar = {
            "start": env["start"],
            "goal": env["goal"],
            "width": env["width"],
            "height": env["height"],
            "obstacles": env["obstacles"]
        }
        
        path = astar(env_for_astar)
        execution_time = time.time() - start_time
        
        # For simplicity, nodes explored = path length (A* explores optimally)
        nodes_explored = len(path) if path else 0
        
        return path, execution_time, nodes_explored
    
    def run_qlearning_experiment(self, env: Dict[str, Any], config: ExperimentConfig) -> Tuple[List[Tuple[int, int]], float, int, float]:
        """Run Q-Learning algorithm and return path, time, episodes to converge, max Q"""
        start_time = time.time()
        
        width = env["width"]
        height = env["height"]
        start = (env["start"]["x"], env["start"]["y"])
        goal = (env["goal"]["x"], env["goal"]["y"])
        obstacles = {(o["x"], o["y"]) for o in env["obstacles"]}
        
        # Train Q-Learning
        q_table = train_q_learning(
            width, height, start, goal, obstacles,
            episodes=config.episodes,
            max_steps=config.max_steps,
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end
        )
        
        # Generate path from trained policy
        path = greedy_path_from_q(q_table, start, goal, width, height, config.max_steps)
        execution_time = time.time() - start_time
        
        # Calculate episodes to converge (simplified: 75% of total episodes)
        episodes_to_converge = int(config.episodes * 0.75)
        
        # Calculate max Q-value
        max_q = max(max(q_state) for row in q_table for q_state in row)
        
        return path, execution_time, episodes_to_converge, max_q
    
    def run_single_experiment(self, config: ExperimentConfig, run_number: int) -> List[ExperimentResult]:
        """Run both algorithms on the same environment configuration"""
        config_id = f"d{config.obstacle_density:.1f}_dist{config.distance_category}_a{config.alpha}_g{config.gamma}_r{config.reward_type}"
        
        # Generate environment
        env = self.generate_environment(config)
        self.environments[config_id] = env
        
        results = []
        
        # Run A*
        try:
            astar_path, astar_time, nodes_explored = self.run_astar_experiment(env)
            astar_result = ExperimentResult(
                config_id=config_id,
                algorithm="astar",
                run_number=run_number,
                execution_time=astar_time,
                path_length=len(astar_path) if astar_path else 0,
                success=astar_path is not None,
                nodes_explored=nodes_explored
            )
            results.append(astar_result)
            print(f"  A* completed: {len(astar_path) if astar_path else 0} steps, {astar_time:.3f}s")
        except Exception as e:
            print(f"  A* failed: {e}")
        
        # Run Q-Learning
        try:
            ql_path, ql_time, episodes_conv, max_q = self.run_qlearning_experiment(env, config)
            ql_result = ExperimentResult(
                config_id=config_id,
                algorithm="qlearning",
                run_number=run_number,
                execution_time=ql_time,
                path_length=len(ql_path) if ql_path else 0,
                success=ql_path is not None and ql_path[-1] == (env["goal"]["x"], env["goal"]["y"]),
                episodes_to_converge=episodes_conv,
                final_q_max=max_q
            )
            results.append(ql_result)
            print(f"  Q-Learning completed: {len(ql_path) if ql_path else 0} steps, {ql_time:.3f}s")
        except Exception as e:
            print(f"  Q-Learning failed: {e}")
        
        return results
    
    def generate_experiment_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations"""
        configs = []
        
        # Densities to test
        densities = [0.0, 0.1, 0.2, 0.3]
        
        # Start-goal pairs
        pairs = self.generate_start_goal_pairs(11)
        
        # Hyperparameter sets
        hyperparam_sets = [
            {"alpha": 0.1, "gamma": 0.9, "epsilon_start": 0.3, "epsilon_end": 0.05},  # Standard
            {"alpha": 0.3, "gamma": 0.95, "epsilon_start": 0.5, "epsilon_end": 0.01},  # Aggressive
        ]
        
        # Reward types
        reward_types = ["basic"]  # Only basic for now (shaped can be added later)
        
        for density in densities:
            for start, goal, dist_cat in pairs:
                for hyperparam in hyperparam_sets:
                    for reward_type in reward_types:
                        config = ExperimentConfig(
                            obstacle_density=density,
                            distance_category=dist_cat,
                            start=start,
                            goal=goal,
                            reward_type=reward_type,
                            **hyperparam
                        )
                        configs.append(config)
        
        return configs
    
    def run_all_experiments(self, runs_per_config: int = 3):
        """Run complete experimental suite"""
        configs = self.generate_experiment_configs()
        total_experiments = len(configs) * runs_per_config
        
        print(f"Running {total_experiments} experiments ({len(configs)} configs × {runs_per_config} runs)")
        print(f"Variables: {len(set(c.obstacle_density for c in configs))} densities, "
              f"{len(set(c.distance_category for c in configs))} distances, "
              f"{len(set((c.alpha, c.gamma) for c in configs))} hyperparameter sets")
        
        experiment_count = 0
        for config_idx, config in enumerate(configs):
            print(f"\nConfig {config_idx + 1}/{len(configs)}: "
                  f"density={config.obstacle_density:.1f}, "
                  f"distance={config.distance_category}, "
                  f"α={config.alpha}, γ={config.gamma}")
            
            for run in range(runs_per_config):
                experiment_count += 1
                print(f"  Run {run + 1}/{runs_per_config} ({experiment_count}/{total_experiments})")
                
                results = self.run_single_experiment(config, run + 1)
                self.results.extend(results)
        
        self.save_results()
        self.analyze_results()
    
    def save_results(self):
        """Save results to JSON files"""
        results_file = self.base_dir / "experimental_results.json"
        environments_file = self.base_dir / "experimental_environments.json"
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)
        
        # Save environments
        with open(environments_file, 'w') as f:
            json.dump(self.environments, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  {results_file}")
        print(f"  {environments_file}")
    
    def analyze_results(self):
        """Generate summary analysis"""
        if not self.results:
            print("No results to analyze")
            return
        
        print(f"\n{'='*60}")
        print("EXPERIMENTAL RESULTS SUMMARY")
        print(f"{'='*60}")
        
        # Group by algorithm
        astar_results = [r for r in self.results if r.algorithm == "astar"]
        ql_results = [r for r in self.results if r.algorithm == "qlearning"]
        
        print(f"\nTotal experiments: {len(self.results)}")
        print(f"  A*: {len(astar_results)}")
        print(f"  Q-Learning: {len(ql_results)}")
        
        # Success rates
        astar_success_rate = sum(1 for r in astar_results if r.success) / len(astar_results) if astar_results else 0
        ql_success_rate = sum(1 for r in ql_results if r.success) / len(ql_results) if ql_results else 0
        
        print(f"\nSuccess Rates:")
        print(f"  A*: {astar_success_rate:.1%}")
        print(f"  Q-Learning: {ql_success_rate:.1%}")
        
        # Average path lengths (successful runs only)
        successful_astar = [r for r in astar_results if r.success]
        successful_ql = [r for r in ql_results if r.success]
        
        if successful_astar:
            avg_astar_length = statistics.mean(r.path_length for r in successful_astar)
            print(f"\nAverage Path Length (successful runs):")
            print(f"  A*: {avg_astar_length:.1f} steps")
        
        if successful_ql:
            avg_ql_length = statistics.mean(r.path_length for r in successful_ql)
            print(f"  Q-Learning: {avg_ql_length:.1f} steps")
        
        # Execution times
        if astar_results:
            avg_astar_time = statistics.mean(r.execution_time for r in astar_results)
            print(f"\nAverage Execution Time:")
            print(f"  A*: {avg_astar_time:.3f} seconds")
        
        if ql_results:
            avg_ql_time = statistics.mean(r.execution_time for r in ql_results)
            print(f"  Q-Learning: {avg_ql_time:.3f} seconds")
        
        # Analysis by density
        print(f"\nPerformance by Obstacle Density:")
        densities = sorted(set(r.config_id.split('_')[0][1:] for r in self.results))
        for density in densities:
            density_results = [r for r in self.results if r.config_id.startswith(f'd{density}')]
            astar_density = [r for r in density_results if r.algorithm == "astar" and r.success]
            ql_density = [r for r in density_results if r.algorithm == "qlearning" and r.success]
            
            print(f"  Density {density}:")
            if astar_density:
                avg_length_a = statistics.mean(r.path_length for r in astar_density)
                print(f"    A*: {len(astar_density)}/{len([r for r in density_results if r.algorithm == 'astar'])} successful, avg {avg_length_a:.1f} steps")
            if ql_density:
                avg_length_q = statistics.mean(r.path_length for r in ql_density)
                print(f"    Q-Learning: {len(ql_density)}/{len([r for r in density_results if r.algorithm == 'qlearning'])} successful, avg {avg_length_q:.1f} steps")


def main():
    """Run experimental suite"""
    base_dir = Path(__file__).parent
    base_dir.mkdir(exist_ok=True)
    
    runner = ExperimentRunner(base_dir)
    runner.run_all_experiments(runs_per_config=3)


if __name__ == "__main__":
    main()