# Experimental Framework: A* vs Q-Learning Comparison

## Overview

This experimental framework systematically compares A* and Q-Learning algorithms across multiple variables to understand their relative performance characteristics.

## Experimental Design

### Variables Tested (3 primary + 1 secondary)

1. **Obstacle Density**: 0%, 10%, 20%, 30%
2. **Start-Goal Distance**: 
   - Close (2-4 units): quick paths
   - Medium (5-8 units): moderate complexity  
   - Far (9-15 units): long traversals
3. **Q-Learning Hyperparameters**:
   - Standard: α=0.1, γ=0.9, ε=0.3→0.05
   - Aggressive: α=0.3, γ=0.95, ε=0.5→0.01
4. **Reward Type**: Basic rewards (-1/+10/-10) vs Shaped rewards (future enhancement)

### Experimental Combinatorics

- **Base combinations**: 4 densities × 3 distances × 2 hyperparameter sets = 24 configurations
- **Repetitions**: 3 runs per configuration = 72 total experiments per algorithm
- **Total experiments**: 144 (72 A* + 72 Q-Learning)

### Metrics Measured

#### A* Metrics
- Execution time
- Path length (optimal)
- Success rate (should be 100% unless no path exists)
- Nodes explored (approximated)

#### Q-Learning Metrics  
- Training time
- Final path length
- Success rate
- Episodes to convergence (estimated)
- Maximum Q-value achieved

## Usage

### 1. Run Experiments

```bash
cd experiments
python experiment_runner.py
```

This will:
- Generate 24 different environment configurations
- Run both A* and Q-Learning on each configuration  
- Repeat each configuration 3 times
- Save results to `experimental_results.json`
- Save environment configs to `experimental_environments.json`

### 2. Analyze Results

```bash
python analyze_results.py
```

This will:
- Generate detailed text analysis (`experimental_analysis.txt`)
- Create CSV summary for spreadsheet analysis (`experimental_summary.csv`)
- Print quick summary to console

## Expected Results & Hypotheses

### Hypothesis 1: Path Optimality
- **A*** should consistently find optimal paths
- **Q-Learning** may find suboptimal paths, especially with higher obstacle density

### Hypothesis 2: Execution Time
- **A*** should be faster for single path finding
- **Q-Learning** training time should scale with environment complexity

### Hypothesis 3: Robustness to Obstacles
- **A*** performance should be relatively stable across densities
- **Q-Learning** should degrade more significantly with higher obstacle density

### Hypothesis 4: Distance Sensitivity
- Both algorithms should handle longer distances, but Q-Learning may require more episodes to converge

### Hypothesis 5: Hyperparameter Impact
- Higher learning rate (α) should speed convergence but may reduce stability
- Higher discount factor (γ) should improve long-term planning

## File Structure

```
experiments/
├── experiment_runner.py     # Main experimental framework
├── analyze_results.py       # Results analysis and reporting
├── experimental_results.json    # Raw experimental data
├── experimental_environments.json  # Environment configurations used
├── experimental_analysis.txt     # Human-readable analysis report
├── experimental_summary.csv      # Spreadsheet-ready summary
└── README.md               # This file
```

## Customization

### Adding New Variables

To test additional variables, modify `ExperimentRunner.generate_experiment_configs()`:

1. **Grid Size Variation**: Test 5x5, 11x11, 21x21 grids
2. **Reward Shaping**: Implement distance-based intermediate rewards
3. **Episode Limits**: Test different training episode counts
4. **Multi-Agent**: Compare single vs multi-agent scenarios

### Modifying Metrics

Add new metrics by extending the `ExperimentResult` dataclass and updating the measurement code in `run_single_experiment()`.

## Integration with Unity

The experimental results can be loaded into Unity for visualization:

1. Copy environment configs to `unity/environment.json` 
2. Run algorithms to generate `unity/path.json` and `unity/path_qlearning.json`
3. Use Unity interface to visualize different scenarios
4. Compare paths side-by-side

## Statistical Significance

With 3 repetitions per configuration:
- Sufficient for identifying large performance differences
- May need more repetitions for subtle effects
- Consider running 5-10 repetitions for publication-quality results

## Performance Notes

- Total runtime: ~10-30 minutes depending on machine
- A* experiments are very fast (<1s each)
- Q-Learning experiments take 5-30s each depending on complexity
- Most time spent in Q-Learning training phase

## Future Enhancements

1. **Visualization**: Add matplotlib charts for key comparisons
2. **Statistical Testing**: Add significance tests for performance differences
3. **Real-time Analysis**: Stream results during execution
4. **Parameter Optimization**: Automatic hyperparameter tuning for Q-Learning
5. **Multi-objective**: Consider path length vs training time trade-offs