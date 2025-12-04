import math

def minimax(nodo, profundidad, es_maximizador, arbol_juego):
    # Condición de parada: nodo hoja o profundidad máxima
    if profundidad == 0 or nodo not in arbol_juego:
        return nodo  # Retornamos el valor del nodo hoja (heurística)

    if es_maximizador:
        max_eval = -math.inf
        for hijo in arbol_juego[nodo]:
            eval = minimax(hijo, profundidad - 1, False, arbol_juego)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = math.inf
        for hijo in arbol_juego[nodo]:
            eval = minimax(hijo, profundidad - 1, True, arbol_juego)
            min_eval = min(min_eval, eval)
        return min_eval

# Representación del árbol: Clave = Nodo padre, Valor = Lista de hijos (o valores si son hojas)
# Supongamos un juego donde las hojas tienen valores directos y los nodos intermedios son IDs
# Estructura simplificada para el ejemplo:
juego = {
    'Raiz': ['A', 'B'],
    'A': [3, 5],    # Hijos de A con sus valores
    'B': [2, 9]     # Hijos de B con sus valores
}

# Modificamos levemente la función para que lea este diccionario específico
def minimax_demo(nodo, es_max, datos):
    if isinstance(nodo, int): # Es una hoja
        return nodo
    
    if es_max:
        return max([minimax_demo(hijo, False, datos) for hijo in datos[nodo]])
    else:
        return min([minimax_demo(hijo, True, datos) for hijo in datos[nodo]])

print(f"Valor óptimo Mini-Max: {minimax_demo('Raiz', True, juego)}")
