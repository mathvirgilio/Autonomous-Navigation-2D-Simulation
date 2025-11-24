# Simulador de Navegação Autônoma 2D com Desinfecção UV-C

Simulador completo de navegação autônoma para robôs de desinfecção UV-C em ambientes 2D. O projeto implementa algoritmos de navegação baseados em árvore de decisão, simulação de irradiação UV-C, e técnicas de scan matching usando Particle Swarm Optimization (PSO).

## 📋 Índice

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Uso](#uso)
- [Componentes Principais](#componentes-principais)
- [Configuração](#configuração)
- [Exemplos](#exemplos)

## 🚀 Características

- **Navegação Autônoma**: Algoritmo de navegação baseado em árvore de decisão que explora completamente o ambiente
- **Simulação UV-C**: Modelagem física da irradiação UV-C com cálculo de dosagem por área
- **Scan Matching**: Alinhamento de varreduras usando Particle Swarm Optimization
- **Visualização em Tempo Real**: Interface gráfica usando Pygame para visualizar a navegação e desinfecção
- **Mapeamento de Ambientes**: Sistema de sensoriamento que detecta obstáculos e caminhos disponíveis

## 📦 Requisitos

- Python 3.7 ou superior
- Dependências listadas em `requirements.txt`

## 🔧 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/Autonomous-Navigation-2D-Simulation.git
cd Autonomous-Navigation-2D-Simulation
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📁 Estrutura do Projeto

```
Autonomous-Navigation-2D-Simulation/
├── src/                    # Código fonte principal
│   ├── __init__.py
│   ├── simulation_alt.py   # Classe principal de simulação
│   ├── navegation.py      # Algoritmo de navegação baseado em árvore
│   ├── simulador.py       # Funções auxiliares de simulação
│   ├── pso.py             # Implementação do Particle Swarm Optimization
│   ├── scan_matching.py   # Algoritmo de scan matching
│   ├── utils.py           # Funções utilitárias
│   └── data_tree.py       # Estrutura de dados em árvore
├── scripts/               # Scripts auxiliares e testes
│   ├── scan_matching_test.py
│   ├── scanner.py
│   ├── show_map.py
│   └── ...
├── assets/                 # Recursos (imagens, mapas)
│   ├── robot.png
│   ├── img_mapas/
│   └── ...
├── mapas/                  # Arquivos de mapa (.txt)
├── results/                # Resultados de experimentos
├── testes/                 # Testes e benchmarks
├── main.py                 # Script principal
├── requirements.txt        # Dependências do projeto
└── README.md              # Este arquivo
```

## 🎮 Uso

### Execução Básica

Execute o script principal para iniciar a simulação:

```bash
python main.py
```

O script `main.py` configura os parâmetros da simulação e executa a navegação autônoma com desinfecção UV-C.

### Parâmetros Configuráveis

No arquivo `main.py`, você pode ajustar os seguintes parâmetros:

```python
# Informações UV-C
necessary_dosage = 16.9  # mJ/cm² - Dosagem necessária para desinfecção
power = 60              # W - Potência da lâmpada UV-C
attenuation = 10        # % - Atenuação da radiação
exposure_time = 60      # segundos - Tempo de exposição em cada ponto

# Configuração do mapa e robô
file_name = 'mapa1.txt'           # Arquivo do mapa
initial_pos = np.array([2, 2])    # Posição inicial do robô
robot_dim = 2                      # Dimensões do robô (2x2 blocos)
```

### Scan Matching

Para executar testes de scan matching:

```bash
python scripts/scan_matching_test.py
```

## 🔬 Componentes Principais

### 1. Simulation (`simulation_alt.py`)

Classe principal que gerencia:
- Leitura e renderização de mapas
- Movimentação do robô
- Cálculo de dosagem UV-C por área
- Visualização em tempo real com Pygame
- Sistema de sensoriamento (radar/lidar simulado)

### 2. Navigation (`navegation.py`)

Implementa o algoritmo de navegação autônoma:
- Criação de nós de decisão baseados em mudanças de direção
- Detecção de dead-ends (fins de linha)
- Geração de rotas de retorno
- Otimização de caminhos com atalhos
- Uso de árvore de dados para rastreamento de posições visitadas

### 3. PSO (`pso.py`)

Implementação do algoritmo Particle Swarm Optimization:
- Otimização de parâmetros para scan matching
- Suporte a modos atrativo/repulsivo
- Controle de diversidade da população
- Limites de velocidade e posição

### 4. Scan Matching (`scan_matching.py`)

Algoritmo para alinhar varreduras de sensores:
- Comparação de rasters usando Dice Score
- Otimização de transformações (x, y, θ) usando PSO
- Visualização de resultados

## ⚙️ Configuração

### Formatos de Mapa

Os mapas são arquivos de texto onde:
- `0` representa área livre
- `1` representa obstáculo/parede
- Cada linha representa uma linha do mapa

Exemplo de mapa (`mapa1.txt`):
```
1111111111
1000000001
1000000001
1000000001
1111111111
```

### Parâmetros de UV-C

O simulador calcula a dosagem UV-C usando a fórmula:

```
dose = (power × attenuation/100) / (4π × R²) × 1000
```

Onde:
- `power`: Potência da lâmpada (W)
- `attenuation`: Percentual de atenuação
- `R`: Distância do robô ao ponto (pixels convertidos para metros)

## 📊 Exemplos

### Exemplo 1: Simulação Básica

```python
import numpy as np
from src import simulation_alt as obj

# Configuração
file_name = 'mapas/mapa1.txt'
initial_pos = np.array([2, 2])
robot_dim = 2
power = 60
necessary_dosage = 16.9
attenuation = 10
exposure_time = 60

# Executar simulação
simulation = obj.simulation()
simulation.create_display(file_name, initial_pos, robot_dim)
simulation.execute_navegation(power, necessary_dosage, attenuation, exposure_time)
```

### Exemplo 2: Scan Matching

```python
from src.scan_matching import ScanMatching
from src.pso import Particle_Swarm_Optimization
import numpy as np

# Configurar PSO
pso = Particle_Swarm_Optimization(
    n_particles=100,
    n_dimensions=3,
    c1=2.05,
    c2=2.05,
    w_initial=0.9,
    w_final=0.1,
    n_iterations=100
)

# Executar scan matching
scan_matching = ScanMatching(pso, input_solution=np.array([3, 3, 3]))
scan_matching.load_scans(distances1, distances2)
scan_matching.run()
scan_matching.plot()
```

## 📈 Resultados

Após a execução, o simulador exibe:
- Área total livre (m²)
- Percentual de área totalmente limpa
- Dosagem média por área (mJ/cm²)
- Tempo real de execução
- Tempo de processamento

## 🧪 Testes

Os testes estão localizados na pasta `testes/`:
- `benchmarks.py`: Funções de benchmark para PSO
- `pso_com_grafico.py`: Visualização de convergência do PSO
- `teste_pso_graf.py`: Gráficos de teste do PSO

## 📝 Notas

- O simulador assume que cada bloco do mapa tem 0.25m × 0.25m
- A visualização usa Pygame e pode ser fechada a qualquer momento
- Os resultados são salvos automaticamente em imagens (screenshot.jpeg, inicial.jpeg)

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📄 Licença

Este projeto está sob licença [especifique a licença].

## 👥 Autores

[Seu nome/equipe]

---

Para mais informações, consulte os comentários no código ou abra uma issue no repositório.
