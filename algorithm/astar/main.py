import json
import math
import asyncio
from heapq import heappush, heappop
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from dotenv import load_dotenv
import os

load_dotenv()

ENV_FILE = "../unity/environment.json"
PATH_FILE = "../unity/path.json"

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
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
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


class PathfinderBehaviour(OneShotBehaviour):
    """Comportamiento que ejecuta A* una vez"""
    
    async def run(self):
        print(f"[{self.agent.name}] Iniciando búsqueda de camino...")
        
        # Cargar entorno
        env = load_env(ENV_FILE)
        print(f"[{self.agent.name}] Entorno cargado: {env['width']}x{env['height']}")
        print(f"[{self.agent.name}] Inicio: ({env['start']['x']}, {env['start']['y']})")
        print(f"[{self.agent.name}] Meta: ({env['goal']['x']}, {env['goal']['y']})")
        print(f"[{self.agent.name}] Obstáculos: {len(env['obstacles'])}")
        
        # Ejecutar A*
        path = astar(env)
        
        if path is None:
            print(f"[{self.agent.name}] No se encontró camino :(")
        else:
            print(f"[{self.agent.name}] Camino encontrado con {len(path)} pasos.")
            save_path(path, PATH_FILE)
            print(f"[{self.agent.name}] Ruta guardada en {PATH_FILE}")
            
            # Mostrar el camino
            print(f"[{self.agent.name}] Camino: {path[:5]}..." if len(path) > 5 else f"[{self.agent.name}] Camino: {path}")
        
        # Detener el agente
        await self.agent.stop()


class PathfinderAgent(Agent):
    """Agente que encuentra caminos usando A*"""
    
    async def setup(self):
        print(f"[{self.name}] Agente iniciado correctamente")
        print(f"[{self.name}] JID: {self.jid}")
        
        # Agregar el comportamiento
        behaviour = PathfinderBehaviour()
        self.add_behaviour(behaviour)


async def main():
    """Función principal para ejecutar el agente"""
    # Obtener credenciales del .env
    jid = os.getenv("JID")
    password = os.getenv("PASSWORD")
    
    if not jid or not password:
        print("Error: JID o PASSWORD no encontrados en .env")
        return
    
    # Crear y ejecutar el agente
    agent = PathfinderAgent(jid, password)
    await agent.start()
    print("Agente PathfinderAgent iniciado. Presiona Ctrl+C para detener.")
    
    # Esperar a que el agente termine
    while agent.is_alive():
        await asyncio.sleep(1)
    
    print("Agente detenido.")


if __name__ == "__main__":
    asyncio.run(main())
