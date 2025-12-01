import json
import random
import asyncio
from pathlib import Path
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
SCRIPT_DIR = Path(__file__).resolve().parent
UNITY_DIR = SCRIPT_DIR.parent / "unity"
ENV_FILE = UNITY_DIR / "environment.json"
Q_TABLE_FILE = UNITY_DIR / "q_table.json"
PATH_FILE = UNITY_DIR / "path_qlearning.json"

ACTIONS = {
    0: (0, 1),   # up
    1: (0, -1),  # down
    2: (-1, 0),  # left
    3: (1, 0),   # right
    4: (-1, 1),  # up-left
    5: (1, 1),   # up-right
    6: (-1, -1), # down-left
    7: (1, -1),  # down-right
}

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "up-left",
    5: "up-right",
    6: "down-left",
    7: "down-right",
}


def load_environment(path=None):
    if path is None:
        path = ENV_FILE
    with open(path, "r", encoding="utf-8") as f:
        env = json.load(f)
    width = int(env["width"])
    height = int(env["height"])
    
    # Handle both list [x, y] and dict {"x": x, "y": y} formats
    start_data = env["start"]
    if isinstance(start_data, dict):
        start = (int(start_data["x"]), int(start_data["y"]))
    else:
        start = tuple(int(x) for x in start_data)
    
    goal_data = env["goal"]
    if isinstance(goal_data, dict):
        goal = (int(goal_data["x"]), int(goal_data["y"]))
    else:
        goal = tuple(int(x) for x in goal_data)
    
    obstacles = set()
    for o in env["obstacles"]:
        if isinstance(o, dict):
            obstacles.add((int(o["x"]), int(o["y"])))
        else:
            obstacles.add(tuple(int(x) for x in o))
    
    return width, height, start, goal, obstacles


def create_q_table(width, height, n_actions=8):
    # Q[x][y][a]
    return [[[0.0 for _ in range(n_actions)] for _ in range(height)]
            for _ in range(width)]


def epsilon_greedy(q_state, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(len(q_state))
    # greedy
    max_value = max(q_state)
    # si hay empate, elige una de las mejores al azar
    best_actions = [a for a, v in enumerate(q_state) if v == max_value]
    return random.choice(best_actions)


def step(state, action, width, height, goal, obstacles):
    x, y = state
    dx, dy = ACTIONS[action]
    nx, ny = x + dx, y + dy

    # intenta movimiento
    if nx < 0 or nx >= width or ny < 0 or ny >= height or (nx, ny) in obstacles:
        # movimiento inválido: castigo y no se mueve
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


def train_q_learning(
    width,
    height,
    start,
    goal,
    obstacles,
    episodes=2000,
    max_steps=200,
    alpha=0.1,
    gamma=0.9,
    epsilon_start=0.3,
    epsilon_end=0.05,
):
    q_table = create_q_table(width, height)
    epsilon = epsilon_start

    for ep in range(episodes):
        state = start
        for _ in range(max_steps):
            x, y = state
            action = epsilon_greedy(q_table[x][y], epsilon)

            next_state, reward, done = step(state, action, width, height, goal, obstacles)
            nx, ny = next_state

            max_next = max(q_table[nx][ny])
            q_old = q_table[x][y][action]
            q_table[x][y][action] = q_old + alpha * (
                reward + gamma * max_next - q_old
            )

            state = next_state

            if done:
                break

        # decaimiento lineal sencillo de epsilon
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (ep / episodes),
        )

    return q_table


def save_q_table(q_table, path="q_table.json"):
    width = len(q_table)
    height = len(q_table[0])

    q_dict = {}
    for x in range(width):
        for y in range(height):
            state_key = f"{x},{y}"
            q_state = q_table[x][y]
            q_dict[state_key] = {
                ACTION_NAMES[a]: q_state[a]
                for a in range(len(q_state))
            }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(q_dict, f, indent=2)


def greedy_path_from_q(q_table, start, goal, width, height, max_steps=200):
    path = [start]
    state = start
    visited = set()

    for _ in range(max_steps):
        if state == goal:
            break

        x, y = state
        q_state = q_table[x][y]
        # acción greedy pura
        best_action = q_state.index(max(q_state))
        dx, dy = ACTIONS[best_action]
        next_state = (x + dx, y + dy)

        # evitar bucles tontos
        if next_state in visited:
            break

        visited.add(next_state)
        path.append(next_state)
        state = next_state

        if state == goal:
            break

    return path


def save_path(path, path_file="path_qlearning.json"):
    data = {
        "path": [{"x": x, "y": y} for (x, y) in path]
    }
    with open(path_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class QLearningBehaviour(OneShotBehaviour):
    async def run(self):
        print("Q-Learning Agent iniciando entrenamiento...")
        
        # Load environment
        width, height, start, goal, obstacles = load_environment()
        
        # Train Q-learning
        q_table = train_q_learning(
            width,
            height,
            start,
            goal,
            obstacles,
            episodes=2000,
            max_steps=200,
            alpha=0.1,
            gamma=0.9,
            epsilon_start=0.3,
            epsilon_end=0.05,
        )
        
        # Save results
        save_q_table(q_table, Q_TABLE_FILE)
        path = greedy_path_from_q(q_table, start, goal, width, height, max_steps=200)
        save_path(path, PATH_FILE)
        
        print(f"Entrenamiento completado. Ruta guardada con {len(path)} pasos.")
        print(f"Q-table: {Q_TABLE_FILE}")
        print(f"Path: {PATH_FILE}")
        
        # Send completion message to Unity (if needed)
        msg = Message(to="unity@localhost")
        msg.body = f"Training completed. Path length: {len(path)}"
        await self.send(msg)


class QLearningAgent(Agent):
    async def setup(self):
        print(f"Agente Q-Learning iniciado: {self.jid}")
        behaviour = QLearningBehaviour()
        self.add_behaviour(behaviour)


async def main():
    # Get credentials from environment
    jid = os.getenv("JID")
    password = os.getenv("PASSWORD")
    
    if not jid or not password:
        print("Error: JID y PASSWORD deben estar definidos en .env")
        return
    
    # Create and start agent
    agent = QLearningAgent(jid, password)
    
    try:
        await agent.start()
        print("Presiona Ctrl+C para detener el agente...")
        
        # Keep the agent running
        while agent.is_alive():
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nDeteniendo agente...")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
