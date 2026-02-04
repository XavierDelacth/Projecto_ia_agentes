# abordagem_b.py - VERSÃO CORRIGIDA FINAL
# Abordagem B: Exploração completa do ambiente (100% células livres)
# Baseline: N agentes BFS colaborativos (MESMA POLÍTICA, SEM ML)

import numpy as np
import random
import time
import warnings
from collections import defaultdict, deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================
# 1. AMBIENTE PARA ABORDAGEM B (SEM TESOUROS)
# ============================================

class EnvironmentB:
    def __init__(self, size=10, bomb_ratio=0.3):
        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        self.original_grid = None
        self.bomb_ratio = bomb_ratio
        self.treasure_count = 0  # SEM TESOUROS na abordagem B
        self.generate_environment()
        
    def generate_environment(self):
        """Gera ambiente SEM tesouros, apenas bombas e células livres"""
        total_cells = self.size * self.size
        all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        # Garantir que (0,0) seja sempre uma célula livre
        safe_start_positions = [(0, 0)]
        remaining_positions = [pos for pos in all_positions if pos not in safe_start_positions]
        
        # Calcular número de bombas
        bomb_count = int((total_cells - len(safe_start_positions)) * self.bomb_ratio)
        bomb_positions = random.sample(remaining_positions, min(bomb_count, len(remaining_positions)))
        
        # Inicializar grid
        for i in range(self.size):
            for j in range(self.size):
                pos = (i, j)
                if pos in bomb_positions:
                    self.grid[i, j] = 'B'
                else:
                    self.grid[i, j] = 'L'
        
        self.original_grid = self.grid.copy()
        
    def get_cell(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x, y]
        return None
    
    def get_neighbors(self, x, y):
        """Retorna vizinhas válidas (apenas horizontal/vertical)"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Apenas 4 direções
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors

# ============================================
# 2. MEMÓRIA COMPARTILHADA PARA ABORDAGEM B
# ============================================

class SharedMemoryB:
    def __init__(self, env_size=10):
        self.explored = set()
        self.bombs_found = set()
        self.agent_positions = {}
        self.agent_status = {}
        self.cell_knowledge = {}
        self.env_size = env_size
        
        # Compatibilidade com GUI (abordagem B não tem tesouros)
        self.treasures_found = set()
        self.treasures_collected = set()
        
        # Inicializar conhecimento
        for i in range(env_size):
            for j in range(env_size):
                self.cell_knowledge[(i, j)] = {
                    'type': 'U',  # U = Unknown
                    'explored': False,
                    'safe': False  # Inicialmente desconhecido (não seguro)
                }
        
        # Marcar posição inicial (0,0) e vizinhas como seguras
        self.cell_knowledge[(0, 0)]['safe'] = True
        # Marcar vizinhas imediatas de (0,0) como seguras também
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 0 + dx, 0 + dy
            if 0 <= nx < env_size and 0 <= ny < env_size:
                self.cell_knowledge[(nx, ny)]['safe'] = True
    
    def update_explored(self, position, content, agent_id, env):
        """Atualiza memória com nova exploração"""
        x, y = position
        self.explored.add(position)
        self.agent_positions[agent_id] = position
        
        # Atualizar conhecimento da célula
        self.cell_knowledge[position]['explored'] = True
        self.cell_knowledge[position]['type'] = content
        
        log_msg = f"Agente {agent_id}: {content} em {position}"
        
        if content == 'B':
            self.bombs_found.add(position)
            self.cell_knowledge[position]['safe'] = False
            log_msg += " (BOMBA)"
        else:
            self.cell_knowledge[position]['safe'] = True
            log_msg += " (seguro)"
            
            # IMPORTANTE: Marcar vizinhas não exploradas como seguras
            neighbors = env.get_neighbors(x, y)
            for neighbor in neighbors:
                if not self.cell_knowledge[neighbor]['explored']:
                    if neighbor not in self.bombs_found:
                        self.cell_knowledge[neighbor]['safe'] = True
            
        return log_msg
    
    def is_safe_cell(self, position):
        """Verifica se célula é segura para visitar"""
        return self.cell_knowledge[position]['safe']
    
    def get_best_unknown_neighbor(self, position, env):
        """Retorna melhor vizinha não explorada"""
        x, y = position
        neighbors = env.get_neighbors(x, y)
        
        # Filtrar apenas células seguras e não exploradas
        safe_unknown = []
        for pos in neighbors:
            if not self.cell_knowledge[pos]['explored'] and self.cell_knowledge[pos]['safe']:
                safe_unknown.append(pos)
        
        return safe_unknown

# ============================================
# 3. AGENTE BFS PARA BASELINE (N AGENTES)
# ============================================

class AgentBFS:
    """
    Agente BFS PURO para baseline
    - Executa APENAS BFS (sem ML, sem inferência)
    - Todos os agentes usam a MESMA política
    - Colaboração apenas via memória compartilhada
    """
    def __init__(self, agent_id, start_pos=(0, 0)):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.steps_taken = 0
        self.treasures_collected = 0  # Sempre 0 para baseline B
        self.bombs_defused = 0
        
        # BFS state
        self.exploration_queue = deque([start_pos])
        self.personal_visited = set([start_pos])
        
        # Direção preferencial baseada no ID (para dividir espaço)
        self.preferred_direction = agent_id % 4  # 0=cima, 1=baixo, 2=esquerda, 3=direita
        
    def choose_action(self, shared_memory, env):
        """
        BFS PURO: escolhe próxima célula da fila
        SEM pesos, SEM ML, SEM decisão inteligente
        """
        if not self.alive:
            return None
        
        x, y = self.position
        
        # 1. Prioridade: vizinhos IMEDIATOS não explorados
        neighbors = env.get_neighbors(x, y)
        unexplored = [n for n in neighbors if n not in shared_memory.explored]
        
        if unexplored:
            # Ordenar por direção preferencial (divisão de espaço)
            unexplored.sort(key=lambda pos: self._direction_priority(pos))
            return unexplored[0]
        
        # 2. Procurar células próximas não exploradas (BFS limitado)
        nearby = self._find_nearby_unexplored(shared_memory, env, radius=3)
        if nearby:
            return nearby[0]
        
        # 3. BFS tradicional: escolher da fila
        while self.exploration_queue:
            candidate = self.exploration_queue.popleft()
            
            if candidate not in shared_memory.explored:
                return candidate
        
        # 4. Expandir fila se necessário
        self._expand_queue(env, shared_memory)
        
        if self.exploration_queue:
            return self.exploration_queue.popleft()
        
        return None
    
    def _direction_priority(self, neighbor):
        """Calcula prioridade baseada na direção preferencial"""
        nx, ny = neighbor
        cx, cy = self.position
        dx, dy = nx - cx, ny - cy
        
        # Mapear direção
        if dx == -1:
            direction = 0  # cima
        elif dx == 1:
            direction = 1  # baixo
        elif dy == -1:
            direction = 2  # esquerda
        else:
            direction = 3  # direita
        
        # Prioridade: menor valor = maior prioridade
        if direction == self.preferred_direction:
            return 0
        return abs(direction - self.preferred_direction) + random.random() * 0.1
    
    def _find_nearby_unexplored(self, shared_memory, env, radius=3):
        """Encontra células não exploradas próximas"""
        x, y = self.position
        candidates = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < env.size and 0 <= ny < env.size:
                    pos = (nx, ny)
                    if pos not in shared_memory.explored:
                        distance = abs(dx) + abs(dy)
                        candidates.append((distance, pos))
        
        candidates.sort(key=lambda x: x[0])
        return [pos for _, pos in candidates]
    
    def _expand_queue(self, env, shared_memory):
        """Expande fila de exploração"""
        x, y = self.position
        neighbors = env.get_neighbors(x, y)
        
        for neighbor in neighbors:
            if neighbor not in self.personal_visited:
                self.personal_visited.add(neighbor)
                self.exploration_queue.append(neighbor)
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posição"""
        if not self.alive or new_position is None:
            return "Agente inativo"
        
        self.position = new_position
        self.steps_taken += 1
        
        x, y = new_position
        cell_content = env.get_cell(x, y)
        log_msg = shared_memory.update_explored(new_position, cell_content, self.id, env)
        
        # Adicionar vizinhos à fila
        neighbors = env.get_neighbors(x, y)
        for neighbor in neighbors:
            if neighbor not in self.personal_visited:
                self.personal_visited.add(neighbor)
                self.exploration_queue.append(neighbor)
        
        # Consequências
        if cell_content == 'B':
            if self.bombs_defused > 0:
                self.bombs_defused -= 1
                log_msg += " | BOMBA DESATIVADA!"
                shared_memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " | AGENTE DESTRUÍDO"
        
        return log_msg
    
    def train_models(self, shared_memory=None, env=None):
        """Método vazio - compatibilidade com GUI"""
        pass

# ============================================
# 4. AGENTE ML PARA GRUPOS (HOMOGÊNEO/HETEROGÊNEO)
# ============================================

class AgentB:
    """Agente com ML para grupos homogêneo/heterogêneo"""
    def __init__(self, agent_id, start_pos=(0, 0), inference_weights=None):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.treasures_collected = 0
        self.bombs_defused = 0
        self.steps_taken = 0
        self.last_action = None
        
        # Inicializar modelos de ML
        self.models = {
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'NaiveBayes': GaussianNB(),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        # Dados para treinamento
        self.training_data = {'features': [], 'labels': []}
        self.models_trained = False
        
        # Motor de inferência
        self.inference_engine = InferenceEngineB(inference_weights)
        
        # Histórico de ações
        self.action_history = deque(maxlen=50)
        
        # Memória individual do agente (para baseline)
        self.memory = SharedMemoryB()
        
        # Inicializar com algum conhecimento básico
        self.initialize_basic_knowledge()
    
    def initialize_basic_knowledge(self):
        """Inicializa conhecimento básico para evitar previsões sem dados"""
        self.training_data['features'].append([0, 0])
        self.training_data['labels'].append('L')
        
        for _ in range(5):
            x, y = random.randint(0, 9), random.randint(0, 9)
            self.training_data['features'].append([x, y])
            self.training_data['labels'].append(random.choice(['L', 'L', 'L', 'B']))
    
    def train_models(self, shared_memory=None, env=None):
        """Treina modelos com dados coletados"""
        if len(self.training_data['features']) >= 10 and not self.models_trained:
            X = np.array(self.training_data['features'])
            y = np.array(self.training_data['labels'])
            
            try:
                for name, model in self.models.items():
                    model.fit(X, y)
                self.models_trained = True
            except:
                pass
    
    def predict_cell(self, cell_position, shared_memory):
        """Preve o tipo de célula usando modelos ou heurística"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        x, y = cell_position
        
        # Se célula já foi explorada, retorna conteúdo conhecido
        if cell_position in memory.explored:
            if cell_position in memory.bombs_found:
                return 'B'
            else:
                return 'L'
        
        # Se modelos estão treinados, usar previsão
        if self.models_trained and len(self.training_data['features']) >= 10:
            predictions = []
            for name, model in self.models.items():
                try:
                    pred = model.predict([[x, y]])[0]
                    predictions.append(pred)
                except:
                    pass
            
            if predictions:
                from collections import Counter
                most_common = Counter(predictions).most_common(1)
                if most_common:
                    return most_common[0][0]
        
        # Heurística baseada na posição
        distance_to_center = np.sqrt((x - 5)**2 + (y - 5)**2)
        if distance_to_center < 3:
            return random.choice(['L', 'L', 'L'])  # Centro mais seguro
        else:
            return random.choice(['L', 'L', 'B', 'L'])  # Bordas mais perigosas
    
    def choose_action(self, shared_memory, env):
        """Escolhe próxima ação com estratégia melhorada"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        if not self.alive:
            return None
        
        x, y = self.position
        
        # Treinar modelos periodicamente
        if self.steps_taken % 20 == 0:
            self.train_models()
        
        # Obter vizinhas
        neighbors = env.get_neighbors(x, y)
        
        # Filtrar apenas células seguras
        safe_neighbors = []
        for neighbor in neighbors:
            if memory.is_safe_cell(neighbor):
                safe_neighbors.append(neighbor)
        
        # Se não houver vizinhas seguras, ficar parado
        if not safe_neighbors:
            return None
        
        # Prever tipo de cada vizinha segura
        available_actions = []
        for neighbor in safe_neighbors:
            # Evitar voltar para onde já esteve recentemente
            if neighbor in self.action_history:
                continue
                
            predicted_type = self.predict_cell(neighbor, memory)
            available_actions.append((neighbor, predicted_type))
        
        # Se todas as ações estão no histórico, limpar histórico
        if not available_actions:
            self.action_history.clear()
            for neighbor in safe_neighbors:
                predicted_type = self.predict_cell(neighbor, memory)
                available_actions.append((neighbor, predicted_type))
        
        # Usar motor de inferência para decidir
        next_pos = self.inference_engine.decide_action(
            available_actions, memory, env, self
        )
        
        return next_pos
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posição"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        if not self.alive or new_position is None:
            return "Agente inativo"
        
        old_pos = self.position
        self.position = new_position
        self.steps_taken += 1
        self.action_history.append(old_pos)
        
        # Explorar nova célula
        x, y = new_position
        cell_content = env.get_cell(x, y)
        
        # Atualizar memória compartilhada
        if shared_memory is None:
            log_msg = memory.update_explored(new_position, cell_content, self.id, env)
        else:
            log_msg = shared_memory.update_explored(new_position, cell_content, self.id, env)
        
        # Para baseline, atualizar também memória individual
        if getattr(self, 'is_baseline', False):
            memory.update_explored(new_position, cell_content, self.id, env)
        
        # Atualizar dados de treinamento
        self.training_data['features'].append([x, y])
        self.training_data['labels'].append(cell_content)
        
        # Consequências da ação
        if cell_content == 'B':
            if self.bombs_defused > 0:
                # Usa poder para desativar bomba
                self.bombs_defused -= 1
                log_msg += " | BOMBA DESATIVADA!"
                if shared_memory is not None:
                    shared_memory.cell_knowledge[new_position]['safe'] = True
                else:
                    memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " | AGENTE DESTRUÍDO"
        
        return log_msg

# ============================================
# 5. MOTOR DE INFERÊNCIA PARA GRUPOS ML
# ============================================

class InferenceEngineB:
    def __init__(self, weights=None):
        self.weights = weights or {
            'L': 2.0,    # Livre - prioridade alta
            'B': -100.0, # Bomba - evitar completamente
            'U': 1.0,    # Desconhecido - explorar
            'E': -0.5    # Explorado - evitar revisitar
        }
    
    def calculate_score(self, cell_type, position, shared_memory, agent):
        """Calcula pontuação considerando vários fatores"""
        base_score = self.weights.get(cell_type, 0.0)
        
        # Penalidade por estar perto de bomba
        x, y = position
        for bx, by in shared_memory.bombs_found:
            distance = abs(bx - x) + abs(by - y)
            if distance == 1:  # Adjacente a bomba
                base_score -= 20.0
        
        # Bonus por exploração de área nova
        if not shared_memory.cell_knowledge[position]['explored']:
            base_score += 3.0
        
        # Penalidade por proximidade a outros agentes (evitar aglomeração)
        for other_id, other_pos in shared_memory.agent_positions.items():
            if other_id != agent.id and other_pos is not None:
                other_distance = abs(x - other_pos[0]) + abs(y - other_pos[1])
                if other_distance <= 2:
                    base_score -= 1.0
        
        return base_score
    
    def decide_action(self, available_cells, shared_memory, env, agent):
        """Decide ação considerando múltiplos fatores"""
        if shared_memory is None or getattr(agent, 'is_baseline', False):
            memory = agent.memory
        else:
            memory = shared_memory
            
        if not available_cells:
            return None
        
        best_score = -float('inf')
        best_action = None
        
        for pos, predicted_type in available_cells:
            # Verificar se é bomba conhecida
            if pos in memory.bombs_found and not memory.cell_knowledge[pos]['safe']:
                score = self.weights['B']
            else:
                score = self.calculate_score(predicted_type, pos, memory, agent)
            
            # Pequena aleatoriedade para evitar deadlocks
            score += random.uniform(-0.1, 0.1)
            
            if score > best_score:
                best_score = score
                best_action = pos
        
        return best_action

# ============================================
# 6. SIMULAÇÃO DA ABORDAGEM B
# ============================================

class ApproachBSimulation:
    def __init__(self, num_agents=4, bomb_ratio=0.3, homogeneous=True, max_steps=500):
        self.env = EnvironmentB(bomb_ratio=bomb_ratio)
        self.shared_memory = SharedMemoryB()
        self.num_agents = num_agents
        self.homogeneous = homogeneous
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        
        # Calcular células livres
        total_cells = self.env.size * self.env.size
        bomb_count = sum(1 for row in self.env.grid for cell in row if cell == 'B')
        self.free_cells = total_cells - bomb_count
        
        self.metrics = {
            'explored_percentage': 0,
            'agents_alive': 0,
            'bombs_triggered': 0,
            'steps_taken': 0,
            'execution_time': 0,
            'success': False,
            'bomb_ratio': bomb_ratio,
            'treasures_found': 0,  # Sempre 0 na abordagem B
            'total_treasures': 0,   # Sempre 0 na abordagem B
            'explored_free_cells': 0
        }
        
        self.setup_agents()
    
    def setup_agents(self):
        """Criar agentes baseado no tipo (homogêneo, heterogêneo ou baseline)"""
        # Marcar posição inicial como explorada ANTES de criar agentes
        initial_pos = (0, 0)
        cell_content = self.env.get_cell(*initial_pos)
        self.shared_memory.explored.add(initial_pos)
        self.shared_memory.cell_knowledge[initial_pos]['explored'] = True
        self.shared_memory.cell_knowledge[initial_pos]['type'] = cell_content
        self.shared_memory.cell_knowledge[initial_pos]['safe'] = True
        
        # Criar agentes - AgentB para homogêneo/heterogêneo
        for i in range(self.num_agents):
            agent = AgentB(agent_id=i, start_pos=initial_pos)
            self.agents.append(agent)

    def run_simulation(self, verbose=False):
        """Executa simulação completa"""
        start_time = time.time()
        step = 0
        
        if verbose:
            self.logs.append(f"=== INÍCIO SIMULAÇÃO ABORDAGEM B ===")
            self.logs.append(f"Agentes: {self.num_agents} | Células livres: {self.free_cells}")
            self.logs.append(f"Bombas: {self.metrics['bomb_ratio']*100}%")
            self.logs.append(f"Tipo: {'Homogêneo' if self.homogeneous else 'Heterogêneo'}")
        
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            step_logs = []
            
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # Escolher e executar ação
                next_pos = agent.choose_action(self.shared_memory, self.env)
                if next_pos:
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
                    if verbose and ("BOMBA" in log_msg):
                        step_logs.append(log_msg)
            
            # Calcular percentagem explorada
            explored_free_cells = 0
            for i in range(self.env.size):
                for j in range(self.env.size):
                    pos = (i, j)
                    if pos in self.shared_memory.explored and self.env.grid[i, j] != 'B':
                        explored_free_cells += 1
            
            explored_pct = (explored_free_cells / self.free_cells * 100) if self.free_cells > 0 else 0
            
            # Atualizar métricas
            self.metrics['explored_percentage'] = explored_pct
            self.metrics['explored_free_cells'] = explored_free_cells
            
            # Atualizar métricas
            if verbose and step % 50 == 0:
                self.logs.append(f"Passo {step}: {explored_pct:.1f}% explorado, {agents_alive} agentes vivos")
            
            # Verificar critério de sucesso: 100% explorado E pelo menos 1 agente vivo
            if explored_pct >= 100.0 and agents_alive > 0:
                self.metrics['success'] = True
                if verbose:
                    self.logs.append(f"✅ SUCESSO! {explored_pct:.1f}% explorado, {agents_alive} agentes vivos")
                break
        
        # Calcular métricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        
        if verbose:
            self.logs.append(f"\n=== FIM DA SIMULAÇÃO ===")
            self.logs.append(f"Tempo: {self.metrics['execution_time']:.2f}s")
            self.logs.append(f"Passos: {self.metrics['steps_taken']}")
            self.logs.append(f"Explorado: {self.metrics['explored_percentage']:.1f}%")
            self.logs.append(f"Agentes vivos: {self.metrics['agents_alive']}")
            self.logs.append(f"Sucesso: {'SIM' if self.metrics['success'] else 'NÃO'}")
        
        return self.metrics
    
    def get_explored_percentage(self):
        """Calcula percentagem de células livres exploradas"""
        explored_free_cells = 0
        for x in range(10):
            for y in range(10):
                pos = (x, y)
                if self.env.grid[x, y] != 'B' and self.shared_memory.cell_knowledge[pos]['explored']:
                    explored_free_cells += 1
        
        return (explored_free_cells / self.free_cells) * 100 if self.free_cells > 0 else 0
    
    def print_logs(self):
        """Exibe logs da simulação"""
        for log in self.logs:
            print(log)

# ============================================
# 7. BASELINE B: N AGENTES BFS COLABORATIVOS
# ============================================

class BaselineB_BFS:
    """
    Baseline B: N agentes BFS colaborativos
    
    REGRAS CRÍTICAS:
    ✅ N agentes permitidos
    ✅ Todos executam o MESMO algoritmo (BFS puro)
    ✅ Colaboração APENAS via memória compartilhada
    ✅ SEM aprendizagem
    ✅ SEM motor de inferência
    ✅ SEM pesos
    ✅ SEM decisão inteligente
    
    Citação para o relatório:
    "Nas baselines, o número de agentes pode variar conforme a configuração 
    da simulação; no entanto, todos os agentes executam o mesmo algoritmo 
    clássico de forma idêntica, sem aprendizagem ou motor de inferência, 
    servindo apenas para explorar o paralelismo e não para introduzir 
    inteligência adicional."
    """
    def __init__(self, num_agents=4, bomb_ratio=0.3, max_steps=500):
        self.env = EnvironmentB(bomb_ratio=bomb_ratio)
        self.shared_memory = SharedMemoryB()
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        
        # Calcular células livres
        total_cells = self.env.size * self.env.size
        bomb_count = sum(1 for row in self.env.grid for cell in row if cell == 'B')
        self.free_cells = total_cells - bomb_count
        
        self.metrics = {
            'explored_percentage': 0,
            'agents_alive': 0,
            'bombs_triggered': 0,
            'steps_taken': 0,
            'execution_time': 0,
            'success': False,
            'bomb_ratio': bomb_ratio,
            'treasures_found': 0,
            'total_treasures': 0,
            'explored_free_cells': 0
        }
        
        # Criar agentes BFS
        self.setup_agents()
    
    def setup_agents(self):
        """
        Criar N agentes BFS
        TODOS executam o MESMO algoritmo
        Diferença: apenas ID e direção preferencial (para dividir espaço)
        """
        # Marcar (0,0) como explorada
        initial_pos = (0, 0)
        cell_content = self.env.get_cell(*initial_pos)
        self.shared_memory.explored.add(initial_pos)
        self.shared_memory.cell_knowledge[initial_pos]['explored'] = True
        self.shared_memory.cell_knowledge[initial_pos]['type'] = cell_content
        self.shared_memory.cell_knowledge[initial_pos]['safe'] = True
        
        # Criar N agentes BFS idênticos
        for i in range(self.num_agents):
            agent = AgentBFS(agent_id=i, start_pos=initial_pos)
            self.agents.append(agent)
    
    def run(self):
        """Executa simulação com N agentes BFS"""
        start_time = time.time()
        step = 0
        
        # Loop principal
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            # Cada agente executa sua ação BFS
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # Escolher próxima célula usando BFS
                next_pos = agent.choose_action(self.shared_memory, self.env)
                
                if next_pos:
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
            
            # Calcular percentagem explorada
            explored_free_cells = 0
            for i in range(self.env.size):
                for j in range(self.env.size):
                    pos = (i, j)
                    if (pos in self.shared_memory.explored and 
                        self.env.grid[i, j] != 'B'):
                        explored_free_cells += 1
            
            explored_pct = (explored_free_cells / self.free_cells * 100) if self.free_cells > 0 else 0
            
            # Atualizar métricas
            self.metrics['explored_percentage'] = explored_pct
            self.metrics['explored_free_cells'] = explored_free_cells
            
            # Verificar sucesso
            if explored_pct >= 100.0 and agents_alive > 0:
                self.metrics['success'] = True
                break
        
        # Calcular métricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        self.metrics['bombs_triggered'] = sum(1 for a in self.agents if not a.alive)
        
        return self.metrics


# ============================================
# 8. FUNÇÕES DE ANÁLISE E COMPARAÇÃO
# ============================================

def run_multiple_simulations_b(num_simulations=5, num_agents=4, homogeneous=True):
    """Executa múltiplas simulações para estatísticas"""
    results = []
    
    for i in range(num_simulations):
        sim = ApproachBSimulation(
            num_agents=num_agents,
            bomb_ratio=0.3,
            homogeneous=homogeneous,
            max_steps=300
        )
        
        metrics = sim.run_simulation(verbose=False)
        results.append(metrics)
    
    # Calcular médias
    avg_explored = np.mean([r['explored_percentage'] for r in results])
    avg_time = np.mean([r['execution_time'] for r in results])
    success_rate = np.mean([1 if r['success'] else 0 for r in results])
    avg_survivors = np.mean([r['agents_alive'] for r in results])
    
    return {
        'type': 'Homogêneo' if homogeneous else 'Heterogêneo',
        'num_agents': num_agents,
        'avg_explored': avg_explored,
        'avg_time': avg_time,
        'success_rate': success_rate,
        'avg_survivors': avg_survivors,
        'results': results
    }

def compare_approaches_b():
    """Compara abordagens homogênea, heterogênea e baseline BFS"""
    print("Comparando Abordagem B...")
    
    results = []
    agent_counts = [2, 4, 6, 8]
    
    for num_agents in agent_counts:
        print(f"\nTestando com {num_agents} agentes...")
        
        # Homogêneo
        homo_results = run_multiple_simulations_b(
            num_simulations=5, num_agents=num_agents, homogeneous=True
        )
        
        # Heterogêneo
        hetero_results = run_multiple_simulations_b(
            num_simulations=5, num_agents=num_agents, homogeneous=False
        )
        
        # Baseline BFS com N agentes
        bfs_results = []
        for _ in range(5):
            bfs = BaselineB_BFS(num_agents=num_agents)
            bfs_metrics = bfs.run()
            bfs_results.append(bfs_metrics)
        
        bfs_avg_explored = np.mean([r['explored_percentage'] for r in bfs_results])
        bfs_avg_time = np.mean([r['execution_time'] for r in bfs_results])
        bfs_success_rate = np.mean([1 if r['success'] else 0 for r in bfs_results])
        
        results.append({
            'num_agents': num_agents,
            'homogeneous': homo_results,
            'heterogeneous': hetero_results,
            'bfs': {
                'avg_explored': bfs_avg_explored,
                'avg_time': bfs_avg_time,
                'success_rate': bfs_success_rate,
                'results': bfs_results
            }
        })
        
        print(f"  Homogêneo: {homo_results['success_rate']:.0%} sucesso")
        print(f"  Heterogêneo: {hetero_results['success_rate']:.0%} sucesso")
        print(f"  BFS ({num_agents} agentes): {bfs_success_rate:.0%} sucesso")
    
    return results

# ============================================
# 9. EXECUÇÃO PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("PROJETO IA - ABORDAGEM B: EXPLORAÇÃO COMPLETA")
    print("="*60)
    
    # Teste rápido
    print("\n1. Teste rápido - 4 agentes heterogêneos:")
    sim = ApproachBSimulation(num_agents=4, homogeneous=False, max_steps=300)
    metrics = sim.run_simulation(verbose=True)
    sim.print_logs()
    
    # Teste Baseline BFS com 4 agentes
    print("\n\n2. Teste Baseline BFS - 4 agentes:")
    baseline = BaselineB_BFS(num_agents=4)
    baseline_metrics = baseline.run()
    print(f"Baseline BFS (4 agentes): {'Sucesso' if baseline_metrics['success'] else 'Falha'}")
    print(f"Explorado: {baseline_metrics['explored_percentage']:.1f}%")
    print(f"Passos: {baseline_metrics['steps_taken']}")
    print(f"Tempo: {baseline_metrics['execution_time']:.2f}s")
    
    # Comparação completa
    print("\n\n3. Comparação completa...")
    results = compare_approaches_b()
    
    print("\n\n4. ANÁLISE DOS RESULTADOS:")
    print("-"*50)
    
    best_homo = max(results, key=lambda x: x['homogeneous']['success_rate'])
    best_hetero = max(results, key=lambda x: x['heterogeneous']['success_rate'])
    
    print(f"\nMelhor configuração Homogênea: {best_homo['num_agents']} agentes")
    print(f"  Taxa de sucesso: {best_homo['homogeneous']['success_rate']:.0%}")
    
    print(f"\nMelhor configuração Heterogênea: {best_hetero['num_agents']} agentes")
    print(f"  Taxa de sucesso: {best_hetero['heterogeneous']['success_rate']:.0%}")
    
    print(f"\nBaseline BFS:")
    print(f"  Executa BFS puro com N agentes colaborativos")
    print(f"  MESMA política para todos os agentes")
    print(f"  SEM aprendizagem, SEM inferência, SEM pesos")