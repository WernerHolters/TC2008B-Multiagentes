using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class PathFollower : MonoBehaviour
{
    [Serializable]
    public class PathData
    {
        public List<Coordenadas> path;   // usa tu struct Coordenadas (x, y)
    }

    public float stepTime = 0.3f;   // Tiempo entre pasos

    private List<Coordenadas> path;
    private int currentIndex = 0;
    private bool isPlaying = false;
    private float timer = 0f;
    private string currentPathFile = "path.json";  // Por defecto A*

    private void Start()
    {
        // No cargar automáticamente, esperar a que el usuario elija algoritmo
    }

    private string GetPathFile()
    {
        return Path.Combine(Application.dataPath, "../" + currentPathFile);
    }

    public void LoadPath()
    {
        string fullPath = GetPathFile();
        if (!File.Exists(fullPath))
        {
            Debug.LogError($"Path file not found: {fullPath}");
            return;
        }

        string json = File.ReadAllText(fullPath);
        PathData data = JsonUtility.FromJson<PathData>(json);

        if (data == null || data.path == null || data.path.Count == 0)
        {
            Debug.LogError($"Path file is empty or malformed: {fullPath}");
            return;
        }

        path = data.path;
        currentIndex = 0;
        Debug.Log($"Path loaded with {path.Count} steps from {currentPathFile}");

        // Colocar al agente en la posición inicial
        if (path.Count > 0)
        {
            transform.position = GridToWorld(path[0]);
        }
    }

    // Llamar desde el botón "A Star"
    public void UseAStar()
    {
        StopSimulation();
        currentPathFile = "path.json";
        Debug.Log($"Switching to A* (file: {currentPathFile})");
        LoadPath();
    }

    // Llamar desde el botón "Q Learning"
    public void UseQLearning()
    {
        StopSimulation();
        currentPathFile = "path_qlearning.json";
        Debug.Log($"Switching to Q-Learning (file: {currentPathFile})");
        LoadPath();
    }

    private Vector3 GridToWorld(Coordenadas c)
    {
        GridManager gm = GridManager.Instance;
        if (gm == null)
            return Vector3.zero;

        float tileSize = gm.TileSize;

        // Misma fórmula que SpawnGrid, pero levantando un poco la esfera
        return new Vector3(c.x * tileSize, 0.5f, c.y * tileSize);
    }

    private void Update()
    {
        if (!isPlaying || path == null || path.Count == 0)
            return;

        timer += Time.deltaTime;
        if (timer >= stepTime && currentIndex < path.Count - 1)
        {
            timer = 0f;
            currentIndex++;
            transform.position = GridToWorld(path[currentIndex]);
        }
    }

    // Llamar desde el botón "Iniciar simulación"
    public void StartSimulation()
    {
        if (path == null || path.Count == 0)
        {
            LoadPath();
            if (path == null || path.Count == 0) return;
        }

        string algorithm = currentPathFile == "path.json" ? "A*" : "Q-Learning";
        Debug.Log($"Starting simulation with {algorithm} using file: {currentPathFile} ({path.Count} steps)");

        currentIndex = 0;
        transform.position = GridToWorld(path[0]);
        isPlaying = true;
        timer = 0f;
    }

    // Llamar desde el botón "Detener simulación"
    public void StopSimulation()
    {
        isPlaying = false;
    }
}
