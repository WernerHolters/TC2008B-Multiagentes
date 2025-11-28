import json
import math
from heapq import heappush, heappop

ENV_FILE = "C:\\Users\\Werner\\Documents\\code\\unity\\Aestrella\\environment.json"
PATH_FILE = "C:\\Users\\Werner\\Documents\\code\\unity\\Aestrella\\path.json"

def load_env(path):
    with open(path, "r") as f:
        return json.load(f)

def save_path(path_cells, out_path):
    data = {
        "path": [{"x": x, "y": y} for (x, y) in path_cells]
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

def neighbors(pos, env):
    x, y = pos
    moves = [
        (1, 0), (-1, 0), (0, 1), (0, -1),      # 4 direcciones
        (1, 1), (1, -1), (-1, 1), (-1, -1)     # diagonales
    ]

    width = env["width"]
    height = env["height"]
    obstacles = {(o["x"], o["y"]) for o in env["obstacles"]}

    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
            cost = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
            yield (nx, ny), cost

def heuristic(a, b):
    (x1, y1), (x2, y2) = a, b
    return math.hypot(x1 - x2, y1 - y2)  # Euclidiana

def astar(env):
    start = (env["start"]["x"], env["start"]["y"])
    goal = (env["goal"]["x"], env["goal"]["y"])

    open_heap = []
    heappush(open_heap, (0, start))

    came_from = {}
    g_cost = {start: 0.0}

    while open_heap:
        _, current = heappop(open_heap)

        if current == goal:
            # reconstruir camino
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

if __name__ == "__main__":
    env = load_env(ENV_FILE)
    path = astar(env)

    if path is None:
        print("No se encontró camino :(")
    else:
        print(f"Camino encontrado con {len(path)} pasos.")
        save_path(path, PATH_FILE)
        print(f"Ruta guardada en {PATH_FILE}")
