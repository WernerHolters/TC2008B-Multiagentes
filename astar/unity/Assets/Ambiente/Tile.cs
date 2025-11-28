using UnityEngine;

public class Tile : MonoBehaviour
{
    private Renderer tileRenderer;
    private Color originalColor;

    private Coordenadas coordenadas;

    private bool isObstacle = false;
    private bool isStart = false;
    private bool isGoal = false;

    private void Awake()
    {
        tileRenderer = GetComponent<Renderer>();
        if (tileRenderer != null)
        {
            originalColor = tileRenderer.material.color;
        }
    }

    public void SetCoordinates(int x, int y)
    {
        coordenadas = new Coordenadas(x, y);
    }

    public Coordenadas GetCoordinates()
    {
        return coordenadas;
    }

    public bool IsObstacle() => isObstacle;
    public bool IsStart() => isStart;
    public bool IsGoal() => isGoal;

    public void SetStart(bool value)
    {
        if (value)
        {
            isStart = true;
            isGoal = false;
            isObstacle = false;
        }
        else
        {
            isStart = false;
        }
        UpdateColor();
    }

    public void SetGoal(bool value)
    {
        if (value)
        {
            isGoal = true;
            isStart = false;
            isObstacle = false;
        }
        else
        {
            isGoal = false;
        }
        UpdateColor();
    }

    public void SetObstacle(bool value)
    {
        if (value)
        {
            // Un obstáculo no puede ser start ni goal
            isObstacle = true;
            isStart = false;
            isGoal = false;
        }
        else
        {
            isObstacle = false;
        }
        UpdateColor();
    }

    public void ToggleObstacle()
    {
        SetObstacle(!isObstacle);
    }

    private void UpdateColor()
    {
        if (tileRenderer == null)
            tileRenderer = GetComponent<Renderer>();
        if (tileRenderer == null)
            return;

        if (isStart)
            tileRenderer.material.color = Color.green;
        else if (isGoal)
            tileRenderer.material.color = Color.red;
        else if (isObstacle)
            tileRenderer.material.color = Color.black;
        else
            tileRenderer.material.color = originalColor;
    }

}
