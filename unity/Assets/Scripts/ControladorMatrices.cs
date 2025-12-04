using UnityEngine;

public class ControladorMatrices : MonoBehaviour
{
    [Header("Ejercicio 4: Traslación")]
    public Vector3 trasladoDeseado = new Vector3(2, 0, 0);
    public bool aplicarTraslacion = false;

    [Header("Ejercicio 5: Rotación")]
    public float angulo = 45f;
    public Vector3 ejeRotacion = Vector3.up; // Eje Y por defecto
    public bool aplicarRotacion = false;

    private Vector3 posicionOriginal;

    void Start()
    {
        // Guardamos donde estaba el objeto al darle Play
        posicionOriginal = transform.position;
    }

    void Update()
    {
        // Reiniciamos la posición cada frame para calcular desde cero y que se vea limpio
        Vector3 puntoActual = posicionOriginal;

        // --- SOLUCIÓN EJERCICIO 4 (Traslación) ---
        if (aplicarTraslacion)
        {
            /* En coordenadas homogéneas, una matriz de traslación se ve así:
               | 1  0  0  Tx |
               | 0  1  0  Ty |
               | 0  0  1  Tz |
               | 0  0  0  1  |
            */
            Matrix4x4 matrizTraslacion = Matrix4x4.identity; // Inicia como identidad
            matrizTraslacion.m03 = trasladoDeseado.x; // Tx
            matrizTraslacion.m13 = trasladoDeseado.y; // Ty
            matrizTraslacion.m23 = trasladoDeseado.z; // Tz

            // MultiplyPoint aplica la transformación matemática al punto
            puntoActual = matrizTraslacion.MultiplyPoint(puntoActual);
        }

        // --- SOLUCIÓN EJERCICIO 5 (Rotación) ---
        if (aplicarRotacion)
        {
            /* Unity tiene funciones para crear matrices de rotación usando Cuaterniones internamente,
               pero matemáticamente esto genera la matriz de 4x4 de rotación (Cos, Sin, etc.)
               que pide el ejercicio.
            */
            Quaternion rotacion = Quaternion.AngleAxis(angulo, ejeRotacion);
            Matrix4x4 matrizRotacion = Matrix4x4.Rotate(rotacion);

            // Aplicamos la matriz al punto (que quizás ya fue trasladado)
            puntoActual = matrizRotacion.MultiplyPoint(puntoActual);
        }

        // Asignamos el resultado matemático a la posición real del objeto en Unity
        transform.position = puntoActual;
    }
}
