# Projeto IA - Exploração Colaborativa de Ambientes

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Projeto de IA para exploração colaborativa de ambientes por agentes inteligentes. Compara abordagens homogêneas, heterogêneas e baseline em cenários de coleta de tesouros, exploração completa e localização de bandeiras. Inclui simulações, análise de dados, relatórios e interface gráfica.

## 🚀 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/XavierDelacth/Projecto_ia_agentes.git
   cd Projecto_ia_agentes
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Bibliotecas listadas em `requirements.txt`

## 🎯 Funcionalidades

- **Três Abordagens de Exploração**:
  - **Abordagem A**: Coleta de tesouros evitando bombas
  - **Abordagem B**: Exploração completa do ambiente (BFS)
  - **Abordagem C**: Localização de uma bandeira desconhecida (A*)

- **Configurações de Agentes**:
  - Baseline
  - Homogêneos
  - Heterogêneos

- **Interface Gráfica**: GUI interativa para execução e visualização
- **Análise Comparativa**: Métricas e visualizações de desempenho
- **Geração de Relatórios**: Relatórios automáticos em Markdown

## 🏗️ Estrutura do Projeto

```
Projecto_ia_agentes/
├── abordagem/              # Implementações das abordagens A, B e C
├── analise/                # Scripts de análise e visualização
├── data/                   # Dados exportados e relatórios
├── simulation_results/     # Resultados das simulações
├── gui.py                  # Interface gráfica principal
├── run_simulations_*.py    # Scripts para executar simulações
├── relatorio.py            # Gerador de relatórios
└── DOCUMENTACAO_PROJETO.md # Documentação completa
```

## ▶️ Como Usar

### Via Interface Gráfica
```bash
python gui.py
```
Execute simulações manualmente e visualize resultados em tempo real.

### Via Scripts de Simulação
```bash
# Executar simulações da Abordagem A
python run_simulations_a.py

# Executar simulações da Abordagem B
python run_simulations_b.py

# Executar simulações da Abordagem C
python run_simulations_c.py
```

### Análise de Dados
```bash
python analise_dados.py
python postprocess_all_results.py
```

### Geração de Relatórios
```bash
python relatorio.py
```

## 📊 Resultados

As simulações geram dados em `simulation_results/all_results.json` e exportações CSV em `data/`. Relatórios detalhados são criados automaticamente.

## 🤝 Contribuição

Contribuições são bem-vindas! Abra uma issue ou envie um pull request.

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 📖 Documentação Completa

Para informações detalhadas, consulte [DOCUMENTACAO_PROJETO.md](DOCUMENTACAO_PROJETO.md).</content>
<parameter name="filePath">c:\Users\hp\3D Objects\Projecto_ia_agentes\README.md