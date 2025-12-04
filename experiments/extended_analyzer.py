"""
Extended Experiment Analysis and Report Generator
===============================================

Generates comprehensive analysis with extended metrics including:
- Memory usage
- Path optimality
- Convergence stability
- Exploration efficiency
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional


class ExtendedExperimentAnalyzer:
    def __init__(self, results_file: Path):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.reactive_results = [r for r in self.results if r['algorithm'] == 'reactive']
        self.astar_results = [r for r in self.results if r['algorithm'] == 'astar']
        self.qlearning_results = [r for r in self.results if r['algorithm'] == 'qlearning']
    
    def generate_comprehensive_report(self) -> str:
        """Generate detailed analysis report with all metrics"""
        report = []
        
        report.append("="*80)
        report.append("ANÁLISIS EXPERIMENTAL COMPLETO: Reactivo vs A* vs Q-LEARNING")
        report.append("Comparación de Paradigmas de Agentes Inteligentes")
        report.append("="*80)
        report.append("")
        
        # === SECTION 1: EXPERIMENT SUMMARY ===
        report.append("1. RESUMEN DEL EXPERIMENTO")
        report.append("-" * 80)
        report.append(f"Total de experimentos ejecutados: {len(self.results)}")
        report.append(f"  - Experimentos Agente Reactivo: {len(self.reactive_results)}")
        report.append(f"  - Experimentos A*: {len(self.astar_results)}")
        report.append(f"  - Experimentos Q-Learning: {len(self.qlearning_results)}")
        report.append("")
        
        # === SECTION 2: SUCCESS RATES ===
        report.append("2. TASAS DE ÉXITO")
        report.append("-" * 80)
        
        reactive_success_count = sum(1 for r in self.reactive_results if r['success'])
        astar_success_count = sum(1 for r in self.astar_results if r['success'])
        ql_success_count = sum(1 for r in self.qlearning_results if r['success'])
        
        reactive_success_rate = reactive_success_count / len(self.reactive_results) if self.reactive_results else 0
        astar_success_rate = astar_success_count / len(self.astar_results) if self.astar_results else 0
        ql_success_rate = ql_success_count / len(self.qlearning_results) if self.qlearning_results else 0
        
        report.append(f"Reactivo: {reactive_success_count}/{len(self.reactive_results)} ({reactive_success_rate:.1%})")
        report.append(f"A*: {astar_success_count}/{len(self.astar_results)} ({astar_success_rate:.1%})")
        report.append(f"Q-Learning: {ql_success_count}/{len(self.qlearning_results)} ({ql_success_rate:.1%})")
        report.append("")
        
        # Filter successful runs for detailed analysis
        successful_reactive = [r for r in self.reactive_results if r['success']]
        successful_astar = [r for r in self.astar_results if r['success']]
        successful_qlearning = [r for r in self.qlearning_results if r['success']]
        
        if not successful_reactive and not successful_astar and not successful_qlearning:
            report.append("No hay suficientes ejecuciones exitosas para análisis detallado.")
            return "\n".join(report)
        
        # === SECTION 3: PERFORMANCE METRICS ===
        report.append("3. MÉTRICAS DE RENDIMIENTO (Solo Ejecuciones Exitosas)")
        report.append("-" * 80)
        
        # Path length comparison
        if successful_reactive and successful_astar and successful_qlearning:
            reactive_avg_length = statistics.mean(r['path_length'] for r in successful_reactive)
            astar_avg_length = statistics.mean(r['path_length'] for r in successful_astar)
            ql_avg_length = statistics.mean(r['path_length'] for r in successful_qlearning)
            
            report.append(f"\n3.1 Longitud del Camino")
            report.append(f"  Reactivo promedio: {reactive_avg_length:.2f} pasos")
            report.append(f"  A* promedio: {astar_avg_length:.2f} pasos")
            report.append(f"  Q-Learning promedio: {ql_avg_length:.2f} pasos")
            report.append(f"  Diferencia Reactivo vs A*: {reactive_avg_length - astar_avg_length:+.2f} pasos")
            report.append(f"  Diferencia Q-Learning vs A*: {ql_avg_length - astar_avg_length:+.2f} pasos")
            
            # Execution time
            reactive_avg_time = statistics.mean(r['execution_time'] for r in successful_reactive)
            astar_avg_time = statistics.mean(r['execution_time'] for r in successful_astar)
            ql_avg_time = statistics.mean(r['execution_time'] for r in successful_qlearning)
            
            report.append(f"\n3.2 Tiempo de Ejecución")
            report.append(f"  Reactivo promedio: {reactive_avg_time:.4f} segundos")
            report.append(f"  A* promedio: {astar_avg_time:.4f} segundos")
            report.append(f"  Q-Learning promedio: {ql_avg_time:.4f} segundos")
            times = [("Reactivo", reactive_avg_time), ("A*", astar_avg_time), ("Q-Learning", ql_avg_time)]
            times_sorted = sorted(times, key=lambda x: x[1])
            ranking_str = " > ".join([f"{name} ({time:.4f}s)" for name, time in times_sorted])
            report.append(f"  Ranking velocidad: {ranking_str}")
        
        # Memory usage
        astar_with_memory = [r for r in successful_astar if r.get('memory_usage') is not None]
        ql_with_memory = [r for r in successful_qlearning if r.get('memory_usage') is not None]
        
        if astar_with_memory and ql_with_memory:
            astar_avg_memory = statistics.mean(r['memory_usage'] for r in astar_with_memory)
            ql_avg_memory = statistics.mean(r['memory_usage'] for r in ql_with_memory)
            
            report.append(f"\n3.3 Uso de Memoria")
            report.append(f"  A* promedio: {astar_avg_memory:.2f} MB")
            report.append(f"  Q-Learning promedio: {ql_avg_memory:.2f} MB")
            report.append(f"  Diferencia: {ql_avg_memory - astar_avg_memory:+.2f} MB")
            
            if ql_avg_memory > 0:
                memory_ratio = astar_avg_memory / ql_avg_memory
                report.append(f"  Eficiencia relativa: A* usa {memory_ratio:.2f}x la memoria de Q-Learning")
        
        # Path optimality
        astar_with_opt = [r for r in successful_astar if r.get('path_optimality') is not None]
        ql_with_opt = [r for r in successful_qlearning if r.get('path_optimality') is not None]
        
        if astar_with_opt and ql_with_opt:
            astar_avg_opt = statistics.mean(r['path_optimality'] for r in astar_with_opt)
            ql_avg_opt = statistics.mean(r['path_optimality'] for r in ql_with_opt)
            
            report.append(f"\n3.4 Optimalidad del Camino (1.0 = óptimo)")
            report.append(f"  A* promedio: {astar_avg_opt:.3f}")
            report.append(f"  Q-Learning promedio: {ql_avg_opt:.3f}")
            report.append(f"  Diferencia: {ql_avg_opt - astar_avg_opt:+.3f}")
            
            if astar_avg_opt > 0:
                opt_percentage = (ql_avg_opt / astar_avg_opt) * 100
                report.append(f"  Q-Learning alcanza {opt_percentage:.1f}% de la optimalidad de A*")
        
        # Path smoothness
        astar_with_smooth = [r for r in successful_astar if r.get('path_smoothness') is not None]
        ql_with_smooth = [r for r in successful_qlearning if r.get('path_smoothness') is not None]
        
        if astar_with_smooth and ql_with_smooth:
            astar_avg_smooth = statistics.mean(r['path_smoothness'] for r in astar_with_smooth)
            ql_avg_smooth = statistics.mean(r['path_smoothness'] for r in ql_with_smooth)
            
            report.append(f"\n3.5 Suavidad del Camino (1.0 = sin cambios de dirección)")
            report.append(f"  A* promedio: {astar_avg_smooth:.3f}")
            report.append(f"  Q-Learning promedio: {ql_avg_smooth:.3f}")
            report.append(f"  Diferencia: {ql_avg_smooth - astar_avg_smooth:+.3f}")
        
        report.append("")
        
        # === SECTION 4: Q-LEARNING SPECIFIC METRICS ===
        ql_with_convergence = [r for r in successful_qlearning if r.get('episodes_to_converge') is not None]
        
        if ql_with_convergence:
            report.append("4. MÉTRICAS ESPECÍFICAS DE Q-LEARNING")
            report.append("-" * 80)
            
            avg_episodes = statistics.mean(r['episodes_to_converge'] for r in ql_with_convergence)
            min_episodes = min(r['episodes_to_converge'] for r in ql_with_convergence)
            max_episodes = max(r['episodes_to_converge'] for r in ql_with_convergence)
            
            report.append(f"\n4.1 Convergencia del Aprendizaje")
            report.append(f"  Episodios promedio para convergencia: {avg_episodes:.1f}")
            report.append(f"  Rango: {min_episodes} - {max_episodes} episodios")
            
            ql_with_stability = [r for r in ql_with_convergence if r.get('convergence_stability') is not None]
            if ql_with_stability:
                avg_stability = statistics.mean(r['convergence_stability'] for r in ql_with_stability)
                report.append(f"  Estabilidad de convergencia: {avg_stability:.3f}")
            
            ql_with_efficiency = [r for r in ql_with_convergence if r.get('exploration_efficiency') is not None]
            if ql_with_efficiency:
                avg_efficiency = statistics.mean(r['exploration_efficiency'] for r in ql_with_efficiency)
                report.append(f"  Eficiencia de exploración: {avg_efficiency:.3f} ({avg_efficiency*100:.1f}% episodios exitosos)")
            
            ql_with_qmax = [r for r in ql_with_convergence if r.get('final_q_max') is not None]
            if ql_with_qmax:
                avg_qmax = statistics.mean(r['final_q_max'] for r in ql_with_qmax)
                report.append(f"  Valor Q máximo promedio: {avg_qmax:.2f}")
            
            report.append("")
        
        # === SECTION 5: ANALYSIS BY OBSTACLE DENSITY ===
        report.append("5. ANÁLISIS POR DENSIDAD DE OBSTÁCULOS")
        report.append("-" * 80)
        
        densities = sorted(set(r['config_id'].split('_')[0][1:] for r in self.results))
        
        for density in densities:
            density_results = [r for r in self.results if r['config_id'].startswith(f'd{density}')]
            astar_density = [r for r in density_results if r['algorithm'] == 'astar']
            ql_density = [r for r in density_results if r['algorithm'] == 'qlearning']
            
            astar_success = sum(1 for r in astar_density if r['success'])
            ql_success = sum(1 for r in ql_density if r['success'])
            
            report.append(f"\nDensidad {density} ({float(density)*100:.0f}% de obstáculos):")
            report.append(f"  A*:")
            report.append(f"    Tasa de éxito: {astar_success}/{len(astar_density)} ({astar_success/len(astar_density):.1%})")
            
            astar_successful = [r for r in astar_density if r['success']]
            if astar_successful:
                avg_path = statistics.mean(r['path_length'] for r in astar_successful)
                avg_time = statistics.mean(r['execution_time'] for r in astar_successful)
                report.append(f"    Longitud promedio: {avg_path:.2f} pasos")
                report.append(f"    Tiempo promedio: {avg_time:.4f}s")
            
            report.append(f"  Q-Learning:")
            report.append(f"    Tasa de éxito: {ql_success}/{len(ql_density)} ({ql_success/len(ql_density):.1%})")
            
            ql_successful = [r for r in ql_density if r['success']]
            if ql_successful:
                avg_path = statistics.mean(r['path_length'] for r in ql_successful)
                avg_time = statistics.mean(r['execution_time'] for r in ql_successful)
                report.append(f"    Longitud promedio: {avg_path:.2f} pasos")
                report.append(f"    Tiempo promedio: {avg_time:.4f}s")
                
                ql_with_conv = [r for r in ql_successful if r.get('episodes_to_converge')]
                if ql_with_conv:
                    avg_conv = statistics.mean(r['episodes_to_converge'] for r in ql_with_conv)
                    report.append(f"    Convergencia promedio: {avg_conv:.1f} episodios")
        
        report.append("")
        
        # === SECTION 6: ANALYSIS BY DISTANCE CATEGORY ===
        report.append("6. ANÁLISIS POR CATEGORÍA DE DISTANCIA")
        report.append("-" * 80)
        
        distance_map = {'close': 'Corta', 'medium': 'Media', 'far': 'Larga'}
        distance_categories = ['close', 'medium', 'far']
        
        for dist_cat in distance_categories:
            dist_results = [r for r in self.results if f'dist{dist_cat}' in r['config_id']]
            
            if not dist_results:
                continue
            
            astar_dist = [r for r in dist_results if r['algorithm'] == 'astar']
            ql_dist = [r for r in dist_results if r['algorithm'] == 'qlearning']
            
            astar_success = sum(1 for r in astar_dist if r['success'])
            ql_success = sum(1 for r in ql_dist if r['success'])
            
            report.append(f"\nDistancia {distance_map[dist_cat]}:")
            report.append(f"  A* - Éxitos: {astar_success}/{len(astar_dist)} ({astar_success/len(astar_dist):.1%})")
            report.append(f"  Q-Learning - Éxitos: {ql_success}/{len(ql_dist)} ({ql_success/len(ql_dist):.1%})")
            
            astar_successful = [r for r in astar_dist if r['success']]
            ql_successful = [r for r in ql_dist if r['success']]
            
            if astar_successful and ql_successful:
                astar_avg = statistics.mean(r['path_length'] for r in astar_successful)
                ql_avg = statistics.mean(r['path_length'] for r in ql_successful)
                report.append(f"  Longitud promedio - A*: {astar_avg:.2f}, Q-Learning: {ql_avg:.2f}")
        
        report.append("")
        
        # === SECTION 7: CONCLUSIONS AND RECOMMENDATIONS ===
        report.append("7. CONCLUSIONES Y RECOMENDACIONES")
        report.append("="*80)
        report.append("")
        
        report.append("7.1 CONCLUSIONES PRINCIPALES:")
        report.append("")
        
        # Performance conclusion
        if astar_avg_length <= ql_avg_length * 1.1:  # Within 10%
            report.append("• CALIDAD DEL CAMINO: Q-Learning logra caminos competitivos con A*,")
            report.append("  alcanzando resultados casi óptimos después del entrenamiento.")
        else:
            report.append("• CALIDAD DEL CAMINO: A* produce caminos consistentemente más cortos,")
            report.append("  demostrando su optimalidad garantizada.")
        report.append("")
        
        # Speed conclusion
        if ql_avg_time > astar_avg_time * 2:
            report.append("• VELOCIDAD: A* es significativamente más rápido que Q-Learning.")
            report.append(f"  Factor: {ql_avg_time/astar_avg_time:.1f}x más lento (incluye entrenamiento)")
        report.append("")
        
        # Success rate conclusion
        if abs(astar_success_rate - ql_success_rate) < 0.05:
            report.append("• CONFIABILIDAD: Ambos algoritmos muestran tasas de éxito similares,")
            report.append("  demostrando robustez en diferentes configuraciones de entorno.")
        elif astar_success_rate > ql_success_rate:
            report.append("• CONFIABILIDAD: A* muestra mayor tasa de éxito, especialmente")
            report.append("  en entornos con alta densidad de obstáculos.")
        report.append("")
        
        report.append("7.2 CUÁNDO USAR CADA ALGORITMO:")
        report.append("")
        report.append("USAR A* CUANDO:")
        report.append("  ✓ El entorno es completamente conocido de antemano")
        report.append("  ✓ Se requiere solución óptima garantizada")
        report.append("  ✓ El tiempo de respuesta es crítico")
        report.append("  ✓ Los recursos computacionales son limitados")
        report.append("  ✓ El entorno es estático (no cambia)")
        report.append("")
        report.append("USAR Q-LEARNING CUANDO:")
        report.append("  ✓ El entorno es desconocido inicialmente")
        report.append("  ✓ El entorno puede cambiar dinámicamente")
        report.append("  ✓ Se puede entrenar offline antes de la ejecución")
        report.append("  ✓ Se requiere adaptación a nuevos escenarios")
        report.append("  ✓ La optimalidad absoluta no es crítica")
        report.append("")
        
        report.append("7.3 OPTIMIZACIONES RECOMENDADAS:")
        report.append("")
        report.append("PARA Q-LEARNING:")
        report.append("  1. Ajustar episodios de entrenamiento según complejidad del entorno")
        report.append("     - Entornos simples (densidad < 0.1): ~500-1000 episodios")
        report.append("     - Entornos complejos (densidad > 0.2): ~2000-3000 episodios")
        report.append("")
        report.append("  2. Implementar early stopping basado en estabilidad de convergencia")
        report.append("     - Monitorear varianza de recompensas en ventana de 100 episodios")
        report.append("     - Detener cuando convergencia_stability > 0.95")
        report.append("")
        report.append("  3. Considerar reward shaping para acelerar aprendizaje")
        report.append("     - Recompensas basadas en distancia al objetivo")
        report.append("     - Penalizaciones graduales por exploración ineficiente")
        report.append("")
        report.append("  4. Optimizar hiperparámetros:")
        report.append("     - α (learning rate): 0.1-0.3 dependiendo de complejidad")
        report.append("     - γ (discount factor): 0.9-0.95 para planificación a largo plazo")
        report.append("     - ε (exploration): decay exponencial desde 0.5 hasta 0.01")
        report.append("")
        report.append("PARA A*:")
        report.append("  1. Implementar caching de rutas para entornos estáticos")
        report.append("  2. Considerar variantes como IDA* para grandes espacios de búsqueda")
        report.append("  3. Usar heurísticas admisibles más informadas cuando sea posible")
        report.append("  4. Implementar poda de nodos para mejorar eficiencia")
        report.append("")
        
        report.append("7.4 CONSIDERACIONES PARA SISTEMAS MULTI-AGENTE:")
        report.append("")
        report.append("  • A* puede requerir coordinación centralizada para evitar colisiones")
        report.append("  • Q-Learning permite aprendizaje descentralizado con políticas cooperativas")
        report.append("  • Considerar algoritmos híbridos que combinen planificación y aprendizaje")
        report.append("  • Implementar mecanismos de negociación para resolución de conflictos")
        report.append("")
        
        report.append("="*80)
        report.append("FIN DEL REPORTE")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_report(self, output_path: Path):
        """Save comprehensive report to file"""
        report = self.generate_comprehensive_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Reporte completo guardado en: {output_path}")


def main():
    """Generate extended analysis report"""
    base_dir = Path(__file__).parent
    results_file = base_dir / "experimental_results.json"
    output_file = base_dir / "reporte_completo_experimental.txt"
    
    if not results_file.exists():
        print(f"❌ Archivo de resultados no encontrado: {results_file}")
        print("Ejecuta experiment_runner.py primero para generar los resultados.")
        return
    
    print("🔍 Generando reporte experimental extendido...")
    analyzer = ExtendedExperimentAnalyzer(results_file)
    analyzer.save_report(output_file)
    
    print("\n" + "="*60)
    print("RESUMEN RÁPIDO")
    print("="*60)
    
    total_experiments = len(analyzer.results)
    reactive_success = sum(1 for r in analyzer.reactive_results if r['success'])
    astar_success = sum(1 for r in analyzer.astar_results if r['success'])
    qlearning_success = sum(1 for r in analyzer.qlearning_results if r['success'])
    
    print(f"Total de experimentos: {total_experiments}")
    if analyzer.reactive_results:
        print(f"Tasa de éxito Reactivo: {reactive_success}/{len(analyzer.reactive_results)} ({reactive_success/len(analyzer.reactive_results):.1%})")
    print(f"Tasa de éxito A*: {astar_success}/{len(analyzer.astar_results)} ({astar_success/len(analyzer.astar_results):.1%})")
    print(f"Tasa de éxito Q-Learning: {qlearning_success}/{len(analyzer.qlearning_results)} ({qlearning_success/len(analyzer.qlearning_results):.1%})")
    
    successful_reactive = [r for r in analyzer.reactive_results if r['success']]
    successful_astar = [r for r in analyzer.astar_results if r['success']]
    successful_qlearning = [r for r in analyzer.qlearning_results if r['success']]
    
    if successful_reactive and successful_astar and successful_qlearning:
        reactive_avg_length = statistics.mean(r['path_length'] for r in successful_reactive)
        astar_avg_length = statistics.mean(r['path_length'] for r in successful_astar)
        qlearning_avg_length = statistics.mean(r['path_length'] for r in successful_qlearning)
        
        print(f"\nLongitud promedio del camino:")
        print(f"Reactivo: {reactive_avg_length:.1f} pasos")
        print(f"A*: {astar_avg_length:.1f} pasos")
        print(f"Q-Learning: {qlearning_avg_length:.1f} pasos")


if __name__ == "__main__":
    main()
