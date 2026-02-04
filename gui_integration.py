# gui_integration.py
# Integração do Sistema de Análise Comparativa com GUI Existente
# Adiciona funcionalidade de armazenamento e comparação avançada

import sys
from pathlib import Path

# Importar módulo de análise comparativa
from comparative_analysis import (
    DataStorage, MetricsCalculator, ComparativeAnalyzer,
    ComparisonVisualizer, create_comparison_window
)

# ============================================
# EXTENSÃO DA CLASSE IAProjectGUI
# ============================================

class ComparativeExtension:
    """
    Extensão para adicionar funcionalidade comparativa à GUI existente
    
    USO:
    1. Importar no gui.py
    2. Inicializar no __init__ da GUI
    3. Conectar aos eventos de simulação
    """
    
    def __init__(self, gui_instance):
        self.gui = gui_instance
        
        # Inicializar storage
        self.storage = DataStorage()
        self.calculator = MetricsCalculator()
        
        # Conectar eventos
        self._connect_events()
        
        print(f"✅ Sistema de análise comparativa inicializado")
        print(f"📁 Dados salvos em: {self.storage.storage_dir}")
    
    def _connect_events(self):
        """Conecta eventos da GUI para salvar dados automaticamente"""
        # Substituir método finalize_simulation da GUI
        original_finalize = self.gui.finalize_simulation
        
        def new_finalize(start_time):
            # Executar finalização original
            original_finalize(start_time)
            
            # Salvar dados automaticamente
            self.save_current_simulation()
        
        self.gui.finalize_simulation = new_finalize
    
    def save_current_simulation(self):
        """Salva simulação atual no storage"""
        if not self.gui.current_simulation:
            return
        
        sim = self.gui.current_simulation
        approach = self.gui.approach_var.get()
        group_type = self.gui.group_type.get()
        
        # Calcular métricas específicas
        if approach == 'A':
            metrics = self.calculator.calculate_approach_a_metrics(sim)
        elif approach == 'B':
            metrics = self.calculator.calculate_approach_b_metrics(sim)
        elif approach == 'C':
            metrics = self.calculator.calculate_approach_c_metrics(sim)
        else:
            return
        
        # Parâmetros da simulação
        parameters = {
            'num_agents': int(self.gui.agent_spinbox.get()),
            'bomb_ratio': float(self.gui.bomb_scale.get()) / 100.0,
            'treasure_count': int(self.gui.treasure_spinbox.get()),
            'max_steps': int(self.gui.max_steps_spinbox.get()),
            'homogeneous': (group_type == 'homogeneous')
        }
        
        # Salvar
        result = self.storage.save_result(approach, group_type, metrics, parameters)
        
        # Log
        self.gui.log(f"💾 Simulação salva: {approach}-{group_type}", "INFO")
        
        return result
    
    def open_comparison_window(self):
        """Abre janela de comparação avançada"""
        try:
            create_comparison_window(self.gui.root, self.storage)
            self.gui.log("📊 Janela de comparação aberta", "INFO")
        except Exception as e:
            self.gui.log(f"Erro ao abrir comparação: {str(e)}", "ERROR")
    
    def add_comparison_button(self):
        """Adiciona botão de comparação avançada à GUI"""
        import tkinter as tk
        from tkinter import ttk
        
        # Localizar frame de controle (ajustar conforme necessário)
        # Assumindo que existe um button_frame na GUI
        if hasattr(self.gui, 'compare_button'):
            # Adicionar ao lado do botão de comparação existente
            ttk.Button(
                self.gui.compare_button.master,
                text="📊 COMPARAÇÃO AVANÇADA",
                command=self.open_comparison_window,
                style="Start.TButton"
            ).pack(fill='x', pady=5)
    
    def get_statistics_summary(self):
        """Retorna resumo estatístico para exibição"""
        summary = {}
        
        for approach in ['A', 'B', 'C']:
            results = self.storage.get_results_by_approach(approach)
            
            summary[approach] = {
                'total': len(results),
                'homogeneous': len([r for r in results if r['group_type'] == 'homogeneous']),
                'heterogeneous': len([r for r in results if r['group_type'] == 'heterogeneous']),
                'baseline': len([r for r in results if r['group_type'] == 'baseline'])
            }
        
        return summary

# ============================================
# FUNÇÃO DE INTEGRAÇÃO RÁPIDA
# ============================================

def integrate_comparison_system(gui_instance):
    """
    Integra sistema de comparação na GUI existente
    
    Adiciona:
    - Salvamento automático de simulações
    - Botão de comparação avançada
    - Armazenamento persistente
    
    Uso no gui.py:
        from gui_integration import integrate_comparison_system
        
        class IAProjectGUI:
            def __init__(self, root):
                # ... código existente ...
                
                # Integrar sistema comparativo
                self.comparison_ext = integrate_comparison_system(self)
    """
    extension = ComparativeExtension(gui_instance)
    extension.add_comparison_button()
    
    return extension

# ============================================
# MODIFICAÇÕES SUGERIDAS PARA GUI.PY
# ============================================

INTEGRATION_GUIDE = """
=================================================================
GUIA DE INTEGRAÇÃO: Sistema de Análise Comparativa
=================================================================

1. IMPORTAÇÃO
   No início do gui.py, adicionar:
   
   from gui_integration import integrate_comparison_system

2. INICIALIZAÇÃO
   No método __init__ da classe IAProjectGUI, após setup_ui():
   
   # Integrar sistema de comparação avançada
   self.comparison_ext = integrate_comparison_system(self)

3. SALVAMENTO AUTOMÁTICO
   O sistema já intercepta finalize_simulation() automaticamente.
   Cada simulação concluída será salva com métricas específicas.

4. BOTÃO DE COMPARAÇÃO
   Um novo botão "COMPARAÇÃO AVANÇADA" será adicionado automaticamente.

5. DADOS PERSISTENTES
   Todos os dados ficam salvos em: simulation_results/all_results.json
   Podem ser exportados para CSV a qualquer momento.

=================================================================
ESTRUTURA DE DADOS SALVA
=================================================================

{
  "A": [
    {
      "timestamp": "2025-01-25T10:30:00",
      "approach": "A",
      "group_type": "heterogeneous",
      "parameters": {
        "num_agents": 4,
        "bomb_ratio": 0.3,
        "treasure_count": 10,
        "max_steps": 100
      },
      "metrics": {
        "treasure_percentage": 75.5,
        "treasures_per_second": 0.15,
        "risk_ratio": 0.25,
        "reward_risk_ratio": 3.2,
        "exploration_efficiency": 0.08,
        ...
      }
    },
    ...
  ],
  "B": [...],
  "C": [...]
}

=================================================================
MÉTRICAS CALCULADAS POR ABORDAGEM
=================================================================

ABORDAGEM A (Maximizar Tesouros):
  ✓ treasure_percentage - % de tesouros encontrados
  ✓ treasures_per_second - Tesouros por tempo
  ✓ treasures_per_step - Tesouros por passo
  ✓ risk_ratio - Risco assumido (bombas/agentes)
  ✓ reward_risk_ratio - Razão recompensa/risco
  ✓ exploration_efficiency - Eficiência de exploração
  ✓ avg_steps_to_treasure - Média de passos até tesouro

ABORDAGEM B (Exploração Total):
  ✓ explored_percentage - % do ambiente explorado
  ✓ safe_exploration_rate - Taxa de exploração segura
  ✓ survival_rate - % de agentes sobreviventes
  ✓ bombs_identified - Bombas identificadas
  ✓ safe_decisions - Número de decisões seguras
  ✓ cells_per_second - Células exploradas por tempo
  ✓ safety_coverage_score - Score geral de segurança
  ✓ redundancy_rate - Taxa de redundância

ABORDAGEM C (Otimizar Caminho):
  ✓ min_steps_to_flag - Passos até bandeira (mínimo)
  ✓ avg_steps_to_flag - Passos até bandeira (média)
  ✓ min_path_cost - Custo mínimo do caminho
  ✓ avg_path_cost - Custo médio do caminho
  ✓ optimal_distance - Distância ótima (Manhattan)
  ✓ path_deviation - Desvio do caminho ótimo
  ✓ path_efficiency - Eficiência do caminho (%)
  ✓ overall_score - Score geral (menor = melhor)

=================================================================
VISUALIZAÇÕES DISPONÍVEIS
=================================================================

1. GRÁFICOS COMPARATIVOS
   - Layout 3 colunas (A, B, C)
   - 5 linhas de métricas
   - Comparação visual entre grupos ML

2. TABELA RESUMO
   - Organizada por abordagem
   - Métricas principais
   - Estatísticas descritivas

3. ANÁLISE DETALHADA
   - Texto formatado
   - Comparações entre grupos
   - Recomendações

=================================================================
EXPORTAÇÃO DE DADOS
=================================================================

CSV por abordagem:
  approach_A_20250125_103000.csv
  approach_B_20250125_103000.csv
  approach_C_20250125_103000.csv

CSV completo:
  all_approaches_20250125_103000.csv

JSON persistente:
  simulation_results/all_results.json

=================================================================
"""

def print_integration_guide():
    """Imprime guia de integração"""
    print(INTEGRATION_GUIDE)

# ============================================
# TESTE RÁPIDO
# ============================================

if __name__ == "__main__":
    print_integration_guide()
    
    # Demonstração básica
    storage = DataStorage()
    
    # Exemplo: salvar alguns resultados
    for approach in ['A', 'B', 'C']:
        for group_type in ['homogeneous', 'heterogeneous', 'baseline']:
            metrics = {
                'success_rate': np.random.uniform(0.5, 0.9),
                'execution_time': np.random.uniform(1, 10),
                'agents_alive': np.random.randint(1, 5)
            }
            
            if approach == 'A':
                metrics.update({
                    'treasure_percentage': np.random.uniform(50, 90),
                    'treasures_per_second': np.random.uniform(0.1, 0.3),
                    'risk_ratio': np.random.uniform(0.1, 0.5)
                })
            elif approach == 'B':
                metrics.update({
                    'explored_percentage': np.random.uniform(70, 100),
                    'survival_rate': np.random.uniform(60, 100),
                    'safety_coverage_score': np.random.uniform(40, 80)
                })
            elif approach == 'C':
                metrics.update({
                    'min_steps_to_flag': np.random.randint(20, 60),
                    'path_efficiency': np.random.uniform(60, 95),
                    'flag_found': True
                })
            
            parameters = {
                'num_agents': 4,
                'bomb_ratio': 0.3,
                'treasure_count': 10,
                'max_steps': 100
            }
            
            storage.save_result(approach, group_type, metrics, parameters)
    
    print(f"\n✅ Dados de teste salvos em: {storage.storage_dir}")
    print(f"📊 Total de resultados: {sum(len(v) for v in storage.all_results.values())}")
    
    # Criar análise
    analyzer = ComparativeAnalyzer(storage)
    
    print("\n📈 ANÁLISE GERADA:")
    for approach in ['A', 'B', 'C']:
        analysis = analyzer.analyze_approach(approach)
        if analysis:
            print(f"\nAbordagem {approach}: {analysis['total_simulations']} simulações")
            if 'best_group' in analysis['comparison']:
                print(f"  Melhor grupo: {analysis['comparison']['best_group']}")