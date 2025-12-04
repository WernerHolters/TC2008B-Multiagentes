import heapq

class Grafo:
    def __init__(self, matriz_adyacencia):
        self.matriz = matriz_adyacencia
        self.nodos = len(matriz_adyacencia)

    def obtener_vecinos(self, nodo):
        vecinos = []
        for i, peso in enumerate(self.matriz[nodo]):
            if peso > 0:  # Si hay conexión
                vecinos.append((i, peso))
        return vecinos

    def heuristica(self, nodo_actual, nodo_destino):
        # En un grafo abstracto (sin coordenadas x,y), la heurística suele ser 0 o estimada.
        # Aquí usaremos 0 (se comporta como Dijkstra) o una tabla predefinida si existiera.
        return 0 

    def a_star(self, inicio, destino):
        # Cola de prioridad: (costo_f, costo_g, nodo_actual, camino)
        open_set = [(0, 0, inicio, [inicio])]
        visited = set()

        while open_set:
            f, g, actual, camino = heapq.heappop(open_set)

            if actual == destino:
                return camino, g  # Retorna camino y costo total

            if actual in visited:
                continue
            visited.add(actual)

            for vecino, peso in self.obtener_vecinos(actual):
                if vecino not in visited:
                    nuevo_g = g + peso
                    nuevo_f = nuevo_g + self.heuristica(vecino, destino)
                    heapq.heappush(open_set, (nuevo_f, nuevo_g, vecino, camino + [vecino]))
        
        return None, float('inf')

matriz = [
    [0, 1, 4, 0], # Nodo 0 conecta con 1 y 2
    [1, 0, 0, 2], # Nodo 1 conecta con 0 y 3
    [4, 0, 0, 1], # Nodo 2 conecta con 0 y 3
    [0, 2, 1, 0]  # Nodo 3 conecta con 1 y 2
]

grafo = Grafo(matriz)
ruta, costo = grafo.a_star(0, 3)
print(f"Ruta A*: {ruta} con costo: {costo}")
