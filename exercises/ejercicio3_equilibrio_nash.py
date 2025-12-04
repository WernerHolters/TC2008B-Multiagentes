import numpy as np

def encontrar_nash(matriz_pagos):
    """
    matriz_pagos: Matriz de tuplas (pago_jugador1, pago_jugador2)
    """
    filas, cols = matriz_pagos.shape[:2]
    equilibrios = []

    # Iteramos sobre cada celda para verificar si es un equilibrio
    for i in range(filas):
        for j in range(cols):
            pago_p1, pago_p2 = matriz_pagos[i, j]
            
            # Verificar mejor respuesta para Jugador 1 (cambiando filas 'r')
            # ¿Existe alguna fila 'r' donde P1 gane más manteniendo la columna 'j'?
            mejor_para_p1 = True
            for r in range(filas):
                if matriz_pagos[r, j][0] > pago_p1:
                    mejor_para_p1 = False
                    break
            
            # Verificar mejor respuesta para Jugador 2 (cambiando columnas 'c')
            # ¿Existe alguna columna 'c' donde P2 gane más manteniendo la fila 'i'?
            mejor_para_p2 = True
            for c in range(cols):
                if matriz_pagos[i, c][1] > pago_p2:
                    mejor_para_p2 = False
                    break
            
            if mejor_para_p1 and mejor_para_p2:
                equilibrios.append(((i, j), (pago_p1, pago_p2)))

    return equilibrios

# Definimos la matriz del problema A
# Fila 0: (3,2), (0,1)
# Fila 1: (2,3), (1,0)
# Nota: Numpy necesita objetos para guardar tuplas
A = np.array([
    [(3, 2), (0, 1)],
    [(2, 3), (1, 0)]
], dtype=object)

resultado = encontrar_nash(A)
print("Equilibrios de Nash encontrados (Índice, Pagos):")
for res in resultado:
    print(res)
