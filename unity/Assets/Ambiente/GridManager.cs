using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;

public enum GridInteractionMode
{
    None,
    EditObstacles,
    SetStart,
    SetGoal
}

public class GridManager : MonoBehaviour
{
    public static GridManager Instance { get; private set; }
    public GridInteractionMode InteractionMode { get; private set; } = GridInteractionMode.None;
    
    public Coordenadas StartCoordinate { get; private set; }
    public Coordenadas GoalCoordinate { get; private set; }

    private Tile startTile;
    private Tile goalTile;
    
    [SerializeField] public Coordenadas gridSize;
    [field: SerializeField] public int TileSize { get; private set; }
    [SerializeField] private GameObject tilePrefab;

    // Diccionario para acceder rápido a cada tile por coordenada
    private Dictionary<string, Tile> tiles = new Dictionary<string, Tile>();

    // Modo edición de obstáculos
    public bool EditMode { get; private set; } = false;

    // Ruta del archivo JSON
    private string savePath;

    [Serializable]
    private class ObstaclesData
    {
        public List<Coordenadas> obstacles = new List<Coordenadas>();
    }

    // Para Python
    [Serializable]
    public class EnvironmentData
    {
        public int width;
        public int height;
        public Coordenadas start;
        public Coordenadas goal;
        public List<Coordenadas> obstacles = new List<Coordenadas>();
    }

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject); // opcional
        }
        else
        {
            Destroy(gameObject);
            return;
        }

        savePath = Path.Combine(Application.dataPath, "../obstacles.json");
        Debug.Log($"Ruta del archivo de obstáculos: {savePath}");
    }

    private void Start()
    {
        ClearGrid();
        SpawnGrid();
    }

    [ContextMenu("Spawn Grid")]
    private void SpawnGrid()
    {
        ClearGrid();
        tiles.Clear();

        for (int x = 0; x < gridSize.x; x++)
        {
            for (int y = 0; y < gridSize.y; y++)
            {
                Vector3 tilePosition = new Vector3(x * TileSize, 0, y * TileSize);
                GameObject tileGO = Instantiate(tilePrefab, tilePosition, Quaternion.identity, transform);
                tileGO.name = $"Tile {x}, {y}";

                // Etiquetas (coordenadas visibles)
                Etiquetas etiquetas = tileGO.GetComponent<Etiquetas>();
                if (etiquetas != null)
                {
                    etiquetas.SetCoordinates(x, y);
                }

                // Componente Tile (obstáculos)
                Tile tileComponent = tileGO.GetComponent<Tile>();
                if (tileComponent == null)   //Asegura que exista 
                {
                    tileComponent = tileGO.AddComponent<Tile>();
                }
                tileComponent.SetCoordinates(x, y);

                string key = GetKey(x, y);
                tiles[key] = tileComponent;
            }
        }
    }

    public void HandleTileClick(Tile tile)
    {
        if (tile == null) return;

        var coord = tile.GetCoordinates();

        switch (InteractionMode)
        {
            case GridInteractionMode.EditObstacles:
                // No permitir marcar start/goal como obstáculo
                if (!tile.IsStart() && !tile.IsGoal())
                {
                    tile.ToggleObstacle();
                }
                break;

            case GridInteractionMode.SetStart:
                // Start no puede ser obstáculo
                if (tile.IsObstacle())
                    tile.SetObstacle(false);

                // Quitar start anterior
                if (startTile != null && startTile != tile)
                    startTile.SetStart(false);

                startTile = tile;
                StartCoordinate = coord;
                tile.SetStart(true);

                Debug.Log($"Start fijado en: ({coord.x},{coord.y})");
                break;

            case GridInteractionMode.SetGoal:
                // Goal no puede ser obstáculo
                if (tile.IsObstacle())
                    tile.SetObstacle(false);

                // Quitar goal anterior
                if (goalTile != null && goalTile != tile)
                    goalTile.SetGoal(false);

                goalTile = tile;
                GoalCoordinate = coord;
                tile.SetGoal(true);

                Debug.Log($"Goal fijado en: ({coord.x},{coord.y})");
                break;

            case GridInteractionMode.None:
            default:
                // Sin modo activo: no hacer nada o futuro: selección, info, etc.
                break;
        }
    }

    public void SetModeNone()
    {
        InteractionMode = GridInteractionMode.None;
        Debug.Log("Modo: Ninguno");
    }

    public void SetModeEditObstacles()
    {
        InteractionMode = GridInteractionMode.EditObstacles;
        Debug.Log("Modo: Editar Obstáculos");
    }

    public void SetModeSetStart()
    {
        InteractionMode = GridInteractionMode.SetStart;
        Debug.Log("Modo: Seleccionar Start");
    }

    public void SetModeSetGoal()
    {
        InteractionMode = GridInteractionMode.SetGoal;
        Debug.Log("Modo: Seleccionar Goal");
    }


    private void ClearGrid()
    {
        // Elimina hijos (tiles anteriores)
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            DestroyImmediate(transform.GetChild(i).gameObject);
        }
    }

    private string GetKey(int x, int y)
    {
        return $"{x}_{y}";
    }


    

    // ---------- MODO EDICIÓN ----------

    public void EnableEditMode()
    {
        EditMode = true;
        Debug.Log("EditMode ON: ahora puedes hacer clic en tiles para marcar obstáculos.");
    }

    public void DisableEditMode()
    {
        EditMode = false;
        Debug.Log("EditMode OFF.");
    }

    public void ToggleEditMode()
    {
        EditMode = !EditMode;
        Debug.Log($"EditMode = {EditMode}");
    }

    // ---------- GUARDAR OBSTÁCULOS ----------

    public void SaveObstacles()
    {
        ObstaclesData data = new ObstaclesData();

        foreach (var kvp in tiles)
        {
            Tile tile = kvp.Value;
            if (tile != null && tile.IsObstacle())
            {
                data.obstacles.Add(tile.GetCoordinates());
            }
        }

        string json = JsonUtility.ToJson(data, true);
        File.WriteAllText(savePath, json);
        Debug.Log($"Obstáculos guardados en: {savePath}");
    }

    // ---------- CARGAR OBSTÁCULOS ----------

    public void LoadObstacles()
    {
        if (!File.Exists(savePath))
        {
            Debug.LogWarning("No se encontró archivo de obstáculos para cargar.");
            return;
        }

        string json = File.ReadAllText(savePath);
        ObstaclesData data = JsonUtility.FromJson<ObstaclesData>(json);

        // Limpia obstáculos actuales
        foreach (var kvp in tiles)
        {
            Tile tile = kvp.Value;
            if (tile != null)
            {
                tile.SetObstacle(false);
            }
        }

        // Marca los del archivo
        foreach (var coord in data.obstacles)
        {
            string key = GetKey(coord.x, coord.y);
            if (tiles.TryGetValue(key, out Tile tile))
            {
                tile.SetObstacle(true);
            }
        }

        Debug.Log("Obstáculos cargados desde archivo.");
    }

    private void Update()
    {
        // Solo si hay un modo activo y clic izquierdo
        if (InteractionMode == GridInteractionMode.None)
            return;

        if (Input.GetMouseButtonDown(0))
        {
            Camera cam = Camera.main;
            if (cam == null)
            {
                Debug.LogError("No hay MainCamera en la escena.");
                return;
            }

            Ray ray = cam.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hit, 1000f))
            {
                Tile tile = hit.collider.GetComponent<Tile>();
                if (tile != null)
                {
                    HandleTileClick(tile);
                }
                else
                {
                    Debug.Log("El rayo pegó en algo sin Tile: " + hit.collider.name);
                }
            }
            else
            {
                // Para ver si el rayo no golpea nada
                // Debug.Log("Raycast no golpeó ningún objeto.");
            }
        }
    }
    
    // Para Python
    public void ExportEnvironmentForPython()
    {
        EnvironmentData data = new EnvironmentData();
        data.width = gridSize.x;
        data.height = gridSize.y;
        data.start = StartCoordinate;
        data.goal = GoalCoordinate;

        foreach (var kvp in tiles)
        {
            Tile tile = kvp.Value;
            if (tile != null && tile.IsObstacle())
            {
                data.obstacles.Add(tile.GetCoordinates());
            }
        }

        // Guardar
        string envPath = Path.Combine(Application.dataPath, "../environment.json");
        string json = JsonUtility.ToJson(data, true);
        File.WriteAllText(envPath, json);
        Debug.Log("Environment exportado a: " + envPath);
    }
}
