import json
import random
from pathlib import Path

ACTIONS = {
    0: (0, 1),   # up
    1: (0, -1),  # down
    2: (-1, 0),  # left
    3: (1, 0),   # right
}

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}


def load_environment(path: str = "../unity/environment.json"):
    with open(path, "r", encoding="utf-8") as f:
        env = json.load(f)
    width = env["width"]
    height = env["height"]
    start = tuple(env["start"])
    goal = tuple(env["goal"])
    obstacles = {tuple(o) for o in env["obstacles"]}
    return width, height, start, goal, obstacles


def create_q_table(width, height, n_actions=4):
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
    serializable = [{"x": x, "y": y} for (x, y) in path]
    with open(path_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def main():
    width, height, start, goal, obstacles = load_environment("environment.json")

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

    save_q_table(q_table, "q_table.json")

    # para modo "Run Policy": generas una ruta greedy y Unity la anima
    path = greedy_path_from_q(q_table, start, goal, width, height, max_steps=200)
    save_path(path, "path_qlearning.json")


if __name__ == "__main__":
    main()
