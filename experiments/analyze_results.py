"""
Experiment Analysis and Report Generator
=======================================

Generates detailed analysis and visualizations of A* vs Q-Learning experiments
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ComparisonMetrics:
    """Metrics comparing A* and Q-Learning performance"""
    astar_success_rate: float
    qlearning_success_rate: float
    astar_avg_path_length: float
    qlearning_avg_path_length: float
    astar_avg_time: float
    qlearning_avg_time: float
    path_length_difference: float  # QL - A* (positive means QL is longer)
    time_difference: float         # QL - A* (positive means QL is slower)


class ExperimentAnalyzer:
    def __init__(self, results_file: Path):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.astar_results = [r for r in self.results if r['algorithm'] == 'astar']
        self.qlearning_results = [r for r in self.results if r['algorithm'] == 'qlearning']
    
    def analyze_by_variable(self, variable_key: str) -> Dict[str, ComparisonMetrics]:
        """Analyze performance grouped by a specific variable"""
        groups = {}
        
        for result in self.results:
            # Extract variable value from config_id
            config_parts = result['config_id'].split('_')
            
            if variable_key == "density":
                value = config_parts[0][1:]  # Remove 'd' prefix
            elif variable_key == "distance":
                value = config_parts[1].replace('dist', '')
            elif variable_key == "alpha":
                value = config_parts[2][1:]  # Remove 'a' prefix
            elif variable_key == "gamma":
                value = config_parts[3][1:]  # Remove 'g' prefix
            else:
                continue
            
            if value not in groups:
                groups[value] = {'astar': [], 'qlearning': []}
            
            groups[value][result['algorithm']].append(result)
        
        # Calculate metrics for each group
        metrics = {}
        for value, algorithms in groups.items():
            astar_data = algorithms['astar']
            qlearning_data = algorithms['qlearning']
            
            # Success rates
            astar_success_rate = sum(1 for r in astar_data if r['success']) / len(astar_data) if astar_data else 0
            qlearning_success_rate = sum(1 for r in qlearning_data if r['success']) / len(qlearning_data) if qlearning_data else 0
            
            # Successful runs only
            astar_successful = [r for r in astar_data if r['success']]
            qlearning_successful = [r for r in qlearning_data if r['success']]
            
            # Path lengths
            astar_avg_length = statistics.mean(r['path_length'] for r in astar_successful) if astar_successful else 0
            qlearning_avg_length = statistics.mean(r['path_length'] for r in qlearning_successful) if qlearning_successful else 0
            
            # Execution times
            astar_avg_time = statistics.mean(r['execution_time'] for r in astar_data) if astar_data else 0
            qlearning_avg_time = statistics.mean(r['execution_time'] for r in qlearning_data) if qlearning_data else 0
            
            # Differences
            path_diff = qlearning_avg_length - astar_avg_length
            time_diff = qlearning_avg_time - astar_avg_time
            
            metrics[value] = ComparisonMetrics(
                astar_success_rate=astar_success_rate,
                qlearning_success_rate=qlearning_success_rate,
                astar_avg_path_length=astar_avg_length,
                qlearning_avg_path_length=qlearning_avg_length,
                astar_avg_time=astar_avg_time,
                qlearning_avg_time=qlearning_avg_time,
                path_length_difference=path_diff,
                time_difference=time_diff
            )
        
        return metrics
    
    def generate_report(self) -> str:
        """Generate comprehensive text report"""
        report = []
        report.append("="*80)
        report.append("COMPARACIÓN EXPERIMENTAL: A* vs Q-LEARNING")
        report.append("="*80)
        
        # Overall summary
        total_experiments = len(self.results)
        astar_count = len(self.astar_results)
        qlearning_count = len(self.qlearning_results)
        
        report.append(f"\nRESUMEN DEL EXPERIMENTO")
        report.append("-" * 40)
        report.append(f"Total de experimentos: {total_experiments}")
        report.append(f"Ejecuciones A*: {astar_count}")
        report.append(f"Ejecuciones Q-Learning: {qlearning_count}")
        
        # Success rates
        astar_success = sum(1 for r in self.astar_results if r['success'])
        qlearning_success = sum(1 for r in self.qlearning_results if r['success'])
        
        report.append(f"\nTASAS DE ÉXITO")
        report.append("-" * 40)
        report.append(f"A*: {astar_success}/{astar_count} ({astar_success/astar_count:.1%})")
        report.append(f"Q-Learning: {qlearning_success}/{qlearning_count} ({qlearning_success/qlearning_count:.1%})")
        
        # Performance by obstacle density
        report.append(f"\nRENDIMIENTO POR DENSIDAD DE OBSTÁCULOS")
        report.append("-" * 40)
        density_metrics = self.analyze_by_variable("density")
        
        for density, metrics in sorted(density_metrics.items()):
            report.append(f"\nDensidad {density}:")
            report.append(f"  Tasa de Éxito - A*: {metrics.astar_success_rate:.1%}, Q-Learning: {metrics.qlearning_success_rate:.1%}")
            report.append(f"  Longitud Promedio - A*: {metrics.astar_avg_path_length:.1f}, Q-Learning: {metrics.qlearning_avg_path_length:.1f}")
            report.append(f"  Diferencia de Pasos: {metrics.path_length_difference:+.1f} pasos (Q-Learning vs A*)")
            report.append(f"  Tiempo Promedio - A*: {metrics.astar_avg_time:.3f}s, Q-Learning: {metrics.qlearning_avg_time:.3f}s")
        
        # Performance by distance
        report.append(f"\nRENDIMIENTO POR DISTANCIA INICIO-META")
        report.append("-" * 40)
        distance_metrics = self.analyze_by_variable("distance")
        
        for distance, metrics in distance_metrics.items():
            dist_name = {"close": "Cercana", "medium": "Media", "far": "Lejana"}.get(distance, distance)
            report.append(f"\nDistancia {dist_name}:")
            report.append(f"  Tasa de Éxito - A*: {metrics.astar_success_rate:.1%}, Q-Learning: {metrics.qlearning_success_rate:.1%}")
            report.append(f"  Longitud Promedio - A*: {metrics.astar_avg_path_length:.1f}, Q-Learning: {metrics.qlearning_avg_path_length:.1f}")
            report.append(f"  Diferencia de Pasos: {metrics.path_length_difference:+.1f} pasos (Q-Learning vs A*)")
        
        # Performance by hyperparameters
        report.append(f"\nRENDIMIENTO POR TASA DE APRENDIZAJE (alpha)")
        report.append("-" * 40)
        alpha_metrics = self.analyze_by_variable("alpha")
        
        for alpha, metrics in sorted(alpha_metrics.items()):
            report.append(f"\nalpha = {alpha}:")
            report.append(f"  Tasa de Éxito Q-Learning: {metrics.qlearning_success_rate:.1%}")
            report.append(f"  Longitud Promedio Q-Learning: {metrics.qlearning_avg_path_length:.1f}")
            report.append(f"  Tiempo Promedio Q-Learning: {metrics.qlearning_avg_time:.3f}s")
        
        # Key findings
        report.append(f"\nHALLAZGOS PRINCIPALES")
        report.append("-" * 40)
        
        # Find best and worst performing configurations
        successful_qlearning = [r for r in self.qlearning_results if r['success']]
        if successful_qlearning:
            best_ql = min(successful_qlearning, key=lambda x: x['path_length'])
            worst_ql = max(successful_qlearning, key=lambda x: x['path_length'])
            
            report.append(f"Mejor rendimiento Q-Learning:")
            report.append(f"  Configuración: {best_ql['config_id']}")
            report.append(f"  Longitud del camino: {best_ql['path_length']} pasos")
            report.append(f"  Tiempo: {best_ql['execution_time']:.3f}s")
            
            report.append(f"\nPeor rendimiento Q-Learning:")
            report.append(f"  Configuración: {worst_ql['config_id']}")
            report.append(f"  Longitud del camino: {worst_ql['path_length']} pasos")
            report.append(f"  Tiempo: {worst_ql['execution_time']:.3f}s")
        
        # Summary recommendations
        report.append(f"\nRECOMENDACIONES")
        report.append("-" * 40)
        report.append("1. A* es óptimo para longitud de camino cuando el entorno es completamente conocido")
        report.append("2. Q-Learning requiere ajuste para diferentes complejidades del entorno")
        report.append("3. Mayor densidad de obstáculos degrada el rendimiento de Q-Learning más que A*")
        report.append("4. El tiempo de entrenamiento de Q-Learning aumenta significativamente con la complejidad")
        report.append("5. Considerar enfoques híbridos para aplicaciones en tiempo real")
        
        return "\n".join(report)
    
    def generate_csv_summary(self) -> str:
        """Generate CSV summary for spreadsheet analysis"""
        csv_lines = []
        csv_lines.append("Configuracion,Algoritmo,LongitudPromedioRuta,TiempoPromedio,TasaExito,TotalEjecuciones")
        
        # Group results by config and algorithm
        config_groups = {}
        for result in self.results:
            key = (result['config_id'], result['algorithm'])
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(result)
        
        for (config_id, algorithm), runs in config_groups.items():
            successful_runs = [r for r in runs if r['success']]
            avg_path_length = statistics.mean(r['path_length'] for r in successful_runs) if successful_runs else 0
            avg_time = statistics.mean(r['execution_time'] for r in runs)
            success_rate = len(successful_runs) / len(runs)
            total_runs = len(runs)
            
            csv_lines.append(f"{config_id},{algorithm},{avg_path_length:.1f},{avg_time:.4f},{success_rate:.3f},{total_runs}")
        
        return "\n".join(csv_lines)


def main():
    """Generate analysis report from experimental results"""
    base_dir = Path(__file__).parent
    results_file = base_dir / "experimental_results.json"
    
    if not results_file.exists():
        print(f"Archivo de resultados no encontrado: {results_file}")
        print("Ejecuta experiment_runner.py primero para generar los resultados.")
        return
    
    analyzer = ExperimentAnalyzer(results_file)
    
    # Generate text report
    report = analyzer.generate_report()
    report_file = base_dir / "analisis_experimental.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Reporte de análisis guardado en: {report_file}")
    
    # Generate CSV summary
    csv_summary = analyzer.generate_csv_summary()
    csv_file = base_dir / "resumen_experimental.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(csv_summary)
    
    print(f"Resumen CSV guardado en: {csv_file}")
    
    # Print key findings to console
    print("\n" + "="*60)
    print("RESUMEN RÁPIDO")
    print("="*60)
    
    total_experiments = len(analyzer.results)
    astar_success = sum(1 for r in analyzer.astar_results if r['success'])
    qlearning_success = sum(1 for r in analyzer.qlearning_results if r['success'])
    
    print(f"Total de experimentos: {total_experiments}")
    print(f"Tasa de éxito A*: {astar_success}/{len(analyzer.astar_results)} ({astar_success/len(analyzer.astar_results):.1%})")
    print(f"Tasa de éxito Q-Learning: {qlearning_success}/{len(analyzer.qlearning_results)} ({qlearning_success/len(analyzer.qlearning_results):.1%})")
    
    # Quick performance comparison
    successful_astar = [r for r in analyzer.astar_results if r['success']]
    successful_qlearning = [r for r in analyzer.qlearning_results if r['success']]
    
    if successful_astar and successful_qlearning:
        astar_avg_length = statistics.mean(r['path_length'] for r in successful_astar)
        qlearning_avg_length = statistics.mean(r['path_length'] for r in successful_qlearning)
        
        print(f"\nLongitud promedio del camino:")
        print(f"A*: {astar_avg_length:.1f} pasos")
        print(f"Q-Learning: {qlearning_avg_length:.1f} pasos")
        print(f"Diferencia: {qlearning_avg_length - astar_avg_length:+.1f} pasos (Q-Learning vs A*)")


if __name__ == "__main__":
    main()