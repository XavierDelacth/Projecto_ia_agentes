# abordagem_c.py - MODIFICADO
# Abordagem C: Encontrar a bandeira (DESCONHECIDA) com otimização de caminho
# Baseline ML: Agentes heterogêneos com diferentes algoritmos ML + colaboração

import numpy as np
import random
import time
import warnings
from collections import defaultdict, deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import heapq

warnings.filterwarnings('ignore')

# ============================================
# 1. AMBIENTE PARA ABORDAGEM C (COM BANDEIRA)
# ============================================

class EnvironmentC:
    def __init__(self, size=10, bomb_ratio=0.3, treasure_count=10):
        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        self.original_grid = None
        self.bomb_ratio = bomb_ratio
        self.treasure_count = treasure_count
        self.flag_position = None  # Posição da bandeira (objetivo)
        self.generate_environment()
        
    def generate_environment(self):
        """Gera ambiente com tesouros, bombas e uma bandeira"""
        total_cells = self.size * self.size
        all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        # Garantir que tesouros não sejam mais que 20% do ambiente
        max_treasures = int(total_cells * 0.2)
        self.treasure_count = min(self.treasure_count, max_treasures)
        
        # Escolher posições para tesouros
        treasure_positions = random.sample(all_positions, self.treasure_count)
        
        # Escolher posição para bandeira (não em tesouro)
        remaining_positions = [pos for pos in all_positions if pos not in treasure_positions]
        self.flag_position = random.choice(remaining_positions)
        remaining_positions.remove(self.flag_position)
        
        # Calcular número de bombas
        bomb_count = int((total_cells - self.treasure_count - 1) * self.bomb_ratio)
        bomb_positions = random.sample(remaining_positions, min(bomb_count, len(remaining_positions)))
        
        # Inicializar grid
        for i in range(self.size):
            for j in range(self.size):
                pos = (i, j)
                if pos in treasure_positions:
                    self.grid[i, j] = 'T'
                elif pos in bomb_positions:
                    self.grid[i, j] = 'B'
                elif pos == self.flag_position:
                    self.grid[i, j] = 'F'
                else:
                    self.grid[i, j] = 'L'
        
        self.original_grid = self.grid.copy()
        
    def reset_treasure(self, position):
        """Remove tesouro após ser coletado"""
        x, y = position
        self.grid[x, y] = 'L'
        
    def get_cell(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x, y]
        return None
    
    def get_neighbors(self, x, y):
        """Retorna vizinhas válidas (apenas horizontal/vertical)"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors

# ============================================
# 2. MEMÓRIA COMPARTILHADA PARA ABORDAGEM C
# ============================================

class SharedMemoryC:
    def __init__(self, env_size=10, flag_position=None):
        self.explored = set()
        self.treasures_found = set()
        self.treasures_collected = set()
        self.bombs_found = set()
        self.agent_positions = {}
        self.agent_status = {}
        self.cell_knowledge = {}
        self.env_size = env_size
        self.flag_found = False
        # IMPORTANTE: Bandeira agora é DESCONHECIDA (None até ser descoberta)
        self.flag_position = None  # NÃO MAIS flag_position passado como parâmetro
        self.true_flag_position = flag_position  # Apenas para GUI visualizar (roxo)
        
        # Custos estimados de deslocação (para otimização de caminho)
        self.movement_costs = {}
        
        # Inicializar conhecimento
        for i in range(env_size):
            for j in range(env_size):
                self.cell_knowledge[(i, j)] = {
                    'type': 'U',  # U = Unknown
                    'explored': False,
                    'safe': True,
                    'cost': 1.0,
                    'risk': 0.0
                }
                self.movement_costs[(i, j)] = float('inf')
        
        # Posição inicial (0,0) tem custo 0
        self.movement_costs[(0, 0)] = 0
    
    def update_explored(self, position, content, agent_id, env):
        """Atualiza memória com nova exploração"""
        x, y = position
        self.explored.add(position)
        self.agent_positions[agent_id] = position
        
        # Atualizar conhecimento da célula
        self.cell_knowledge[position]['explored'] = True
        
        log_msg = f"Agente {agent_id}: {content} em {position}"
        
        if content == 'T':
            if position not in self.treasures_collected:
                self.treasures_found.add(position)
                self.treasures_collected.add(position)
                self.cell_knowledge[position]['type'] = 'T'
                env.reset_treasure(position)
                log_msg += " (TESOURO COLETADO!)"
            else:
                self.cell_knowledge[position]['type'] = 'L'
                log_msg += " (LIVRE - Tesouro já coletado)"
        elif content == 'B':
            self.cell_knowledge[position]['type'] = content
            self.bombs_found.add(position)
            self.cell_knowledge[position]['safe'] = False
            self.cell_knowledge[position]['risk'] = 1.0
            self.cell_knowledge[position]['cost'] = float('inf')
            log_msg += " (BOMBA)"
        elif content == 'F':
            # DESCOBERTA DA BANDEIRA!
            self.cell_knowledge[position]['type'] = content
            self.flag_found = True
            self.flag_position = position  # Agora sim, registrar posição descoberta
            log_msg += " (🚩 BANDEIRA ENCONTRADA! OBJETIVO ALCANÇADO!)"
        else:
            self.cell_knowledge[position]['type'] = content
            self.cell_knowledge[position]['safe'] = True
            
        # Atualizar custos das células vizinhas
        self.update_neighbor_costs(position, env)
            
        return log_msg
    
    def update_neighbor_costs(self, position, env):
        """Atualizar custos e riscos das células vizinhas"""
        x, y = position
        neighbors = env.get_neighbors(x, y)
        
        for neighbor in neighbors:
            if not self.cell_knowledge[neighbor]['explored']:
                if position in self.bombs_found:
                    self.cell_knowledge[neighbor]['risk'] += 0.3
                    self.cell_knowledge[neighbor]['cost'] += 0.5
                else:
                    self.cell_knowledge[neighbor]['risk'] = max(0, self.cell_knowledge[neighbor]['risk'] - 0.1)
    
    def is_safe_cell(self, position):
        """Verifica se célula é segura para visitar"""
        return self.cell_knowledge[position]['safe']
    
    def estimate_distance_to_flag(self, position):
        """Estima distância até a bandeira (heurística)"""
        if self.flag_position:
            # Se já encontramos a bandeira, usar distância real
            return abs(position[0] - self.flag_position[0]) + abs(position[1] - self.flag_position[1])
        else:
            # Heurística: exploração uniforme (sem direção preferencial)
            # Priorizar células não exploradas
            return 0  # Retorna 0 para não influenciar decisão

# ============================================
# 3. AGENTE PARA ABORDAGEM C (GRUPOS ML)
# ============================================

class AgentC:
    def __init__(self, agent_id, start_pos=(0, 0), inference_weights=None):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.treasures_collected = 0
        self.bombs_defused = 0
        self.steps_taken = 0
        self.path_cost = 0
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
        
        # Motor de inferência otimizado para busca de objetivo
        self.inference_engine = InferenceEngineC(inference_weights)
        
        # Histórico de ações
        self.action_history = deque(maxlen=50)
        
        # Memória individual do agente
        self.memory = SharedMemoryC()
        
        # Inicializar com algum conhecimento básico
        self.initialize_basic_knowledge()
    
    def initialize_basic_knowledge(self):
        """Inicializa conhecimento básico"""
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
        """Prevê o tipo de célula usando modelos ou heurística"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        x, y = cell_position
        
        # Se célula já foi explorada, retorna conteúdo conhecido
        if cell_position in memory.explored:
            if cell_position in memory.treasures_collected:
                return 'L'
            elif cell_position in memory.bombs_found:
                return 'B'
            elif memory.flag_found and cell_position == memory.flag_position:
                return 'F'
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
        
        # Heurística: exploração uniforme
        return random.choice(['L', 'L', 'L', 'L', 'B'])
    
    def choose_action(self, shared_memory, env):
        """Escolhe próxima ação otimizando exploração para encontrar bandeira"""
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
        
        if not safe_neighbors:
            return None
        
        # Prever tipo de cada vizinha segura
        available_actions = []
        for neighbor in safe_neighbors:
            if neighbor in self.action_history:
                continue
                
            predicted_type = self.predict_cell(neighbor, memory)
            available_actions.append((neighbor, predicted_type))
        
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
        
        # Calcular custo do movimento
        movement_cost = memory.cell_knowledge[new_position]['cost']
        self.path_cost += movement_cost
        
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
        if cell_content == 'T':
            if new_position in (shared_memory.treasures_collected if shared_memory else memory.treasures_collected):
                self.treasures_collected += 1
                self.bombs_defused += 1
                log_msg += f" | Tesouros: {self.treasures_collected}"
        elif cell_content == 'B':
            if self.bombs_defused > 0:
                self.bombs_defused -= 1
                log_msg += " | BOMBA DESATIVADA!"
                if shared_memory is not None:
                    shared_memory.cell_knowledge[new_position]['safe'] = True
                else:
                    memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " | AGENTE DESTRUÍDO"
        elif cell_content == 'F':
            log_msg += f" | CUSTO TOTAL DO CAMINHO: {self.path_cost:.2f}"
        
        return log_msg

# ============================================
# 4. MOTOR DE INFERÊNCIA PARA ABORDAGEM C
# ============================================

class InferenceEngineC:
    def __init__(self, weights=None):
        # Pesos otimizados para busca exploratória (sem conhecer bandeira)
        self.weights = weights or {
            'F': 100.0,   # Bandeira - se descoberta
            'T': 8.0,     # Tesouro - importante para desativar bombas
            'L': 2.0,     # Livre
            'B': -200.0,  # Bomba - evitar completamente
            'U': 3.0,     # Desconhecido - PRIORIZAR exploração
            'E': -0.8     # Explorado - evitar revisitar
        }
        
        self.cost_weight = 0.5
        self.risk_weight = 2.0
        self.exploration_weight = 2.0  # Peso para exploração
    
    def calculate_score(self, cell_type, position, shared_memory, agent):
        """Calcula pontuação otimizada para busca exploratória"""
        base_score = self.weights.get(cell_type, 0.0)
        
        # Se bandeira foi descoberta, priorizar caminho até ela
        if shared_memory.flag_position:
            current_dist = shared_memory.estimate_distance_to_flag(agent.position)
            new_dist = shared_memory.estimate_distance_to_flag(position)
            if new_dist < current_dist:
                base_score += 20.0
        else:
            # Bandeira ainda não descoberta - priorizar exploração
            if not shared_memory.cell_knowledge[position]['explored']:
                base_score += self.exploration_weight * 5.0
        
        # Penalidade por custo e risco
        cost = shared_memory.cell_knowledge[position]['cost']
        risk = shared_memory.cell_knowledge[position]['risk']
        
        base_score -= (cost * self.cost_weight)
        base_score -= (risk * self.risk_weight)
        
        # Penalidade por estar perto de bomba
        x, y = position
        for bx, by in shared_memory.bombs_found:
            bomb_distance = abs(bx - x) + abs(by - y)
            if bomb_distance == 1:
                base_score -= 30.0
        
        return base_score
    
    def decide_action(self, available_cells, shared_memory, env, agent):
        """Decide ação otimizando exploração"""
        if shared_memory is None or getattr(agent, 'is_baseline', False):
            memory = agent.memory
        else:
            memory = shared_memory
            
        if not available_cells:
            return None
        
        best_score = -float('inf')
        best_action = None
        
        for pos, predicted_type in available_cells:
            if pos in memory.bombs_found and not memory.cell_knowledge[pos]['safe']:
                score = self.weights['B']
            else:
                score = self.calculate_score(predicted_type, pos, memory, agent)
            
            # Pequena aleatoriedade
            score += random.uniform(-0.05, 0.05)
            
            if score > best_score:
                best_score = score
                best_action = pos
        
        return best_action

# ============================================
# 5. BASELINE ML - AGENTES HETEROGÊNEOS
# ============================================

class AgentML_Baseline:
    """Agente Baseline ML com algoritmo específico e colaboração"""
    def __init__(self, agent_id, ml_algorithm, start_pos=(0, 0)):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.treasures_collected = 0
        self.bombs_defused = 0
        self.steps_taken = 0
        self.path_cost = 0
        self.ml_algorithm = ml_algorithm  # 'KNN', 'NaiveBayes', 'DecisionTree'
        
        # Inicializar APENAS o modelo específico deste agente
        if ml_algorithm == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=3)
            self.strategy = 'conservative'  # Conservador
        elif ml_algorithm == 'NaiveBayes':
            self.model = GaussianNB()
            self.strategy = 'balanced'  # Equilibrado
        elif ml_algorithm == 'DecisionTree':
            self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
            self.strategy = 'aggressive'  # Agressivo
        else:
            # Fallback para Random Forest
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.strategy = 'balanced'
        
        # Dados para treinamento (individuais)
        self.training_data = {'features': [], 'labels': []}
        self.model_trained = False
        
        # Histórico de ações
        self.action_history = deque(maxlen=50)
        
        # Inicializar conhecimento básico
        self.initialize_basic_knowledge()
        
        # Pesos de decisão baseados na estratégia
        if self.strategy == 'conservative':
            self.weights = {'F': 100.0, 'T': 6.0, 'L': 3.0, 'B': -250.0, 'U': 2.0, 'E': -1.0}
        elif self.strategy == 'aggressive':
            self.weights = {'F': 150.0, 'T': 10.0, 'L': 1.5, 'B': -150.0, 'U': 4.0, 'E': -0.5}
        else:  # balanced
            self.weights = {'F': 120.0, 'T': 8.0, 'L': 2.0, 'B': -200.0, 'U': 3.0, 'E': -0.7}
    
    def initialize_basic_knowledge(self):
        """Inicializa conhecimento básico específico"""
        # Ponto inicial
        self.training_data['features'].append([0, 0])
        self.training_data['labels'].append('L')
        
        # Dados hipotéticos variados por estratégia
        if self.strategy == 'conservative':
            # Mais exemplos de células livres
            for _ in range(7):
                x, y = random.randint(0, 9), random.randint(0, 9)
                self.training_data['features'].append([x, y])
                self.training_data['labels'].append('L')
        elif self.strategy == 'aggressive':
            # Mix de tudo
            for _ in range(5):
                x, y = random.randint(0, 9), random.randint(0, 9)
                self.training_data['features'].append([x, y])
                self.training_data['labels'].append(random.choice(['L', 'B', 'T']))
        else:  # balanced
            for _ in range(6):
                x, y = random.randint(0, 9), random.randint(0, 9)
                self.training_data['features'].append([x, y])
                self.training_data['labels'].append(random.choice(['L', 'L', 'L', 'B']))
    
    def train_model(self, shared_memory=None, env=None):
        """Treina APENAS o modelo deste agente com dados compartilhados"""
        # Combinar dados próprios + dados da memória compartilhada
        all_features = list(self.training_data['features'])
        all_labels = list(self.training_data['labels'])
        
        # Adicionar conhecimento compartilhado
        if shared_memory:
            for pos in shared_memory.explored:
                x, y = pos
                cell_type = shared_memory.cell_knowledge[pos]['type']
                if cell_type != 'U':  # Apenas células conhecidas
                    all_features.append([x, y])
                    all_labels.append(cell_type)
        
        # Treinar se houver dados suficientes
        if len(all_features) >= 10:
            X = np.array(all_features)
            y = np.array(all_labels)
            
            try:
                self.model.fit(X, y)
                self.model_trained = True
            except:
                pass
    
    def predict_cell(self, cell_position, shared_memory):
        """Prevê célula usando APENAS o modelo deste agente"""
        x, y = cell_position
        
        # Se célula já explorada, retornar conhecimento
        if cell_position in shared_memory.explored:
            if cell_position in shared_memory.treasures_collected:
                return 'L'
            elif cell_position in shared_memory.bombs_found:
                return 'B'
            elif shared_memory.flag_found and cell_position == shared_memory.flag_position:
                return 'F'
            else:
                return 'L'
        
        # Usar modelo próprio para prever
        if self.model_trained:
            try:
                pred = self.model.predict([[x, y]])[0]
                return pred
            except:
                pass
        
        # Fallback: heurística baseada na estratégia
        if self.strategy == 'conservative':
            return random.choice(['L', 'L', 'L', 'L', 'B'])
        elif self.strategy == 'aggressive':
            return random.choice(['L', 'L', 'T', 'B'])
        else:
            return random.choice(['L', 'L', 'L', 'B'])
    
    def choose_action(self, shared_memory, env):
        """Escolhe ação baseada no modelo ML + estratégia"""
        if not self.alive:
            return None
        
        x, y = self.position
        
        # Treinar modelo periodicamente com dados compartilhados
        if self.steps_taken % 15 == 0:
            self.train_model(shared_memory, env)
        
        # Obter vizinhas seguras
        neighbors = env.get_neighbors(x, y)
        safe_neighbors = [n for n in neighbors if shared_memory.is_safe_cell(n)]
        
        if not safe_neighbors:
            return None
        
        # Prever tipo de cada vizinha
        available_actions = []
        for neighbor in safe_neighbors:
            if neighbor in self.action_history:
                continue
            predicted_type = self.predict_cell(neighbor, shared_memory)
            available_actions.append((neighbor, predicted_type))
        
        if not available_actions:
            self.action_history.clear()
            for neighbor in safe_neighbors:
                predicted_type = self.predict_cell(neighbor, shared_memory)
                available_actions.append((neighbor, predicted_type))
        
        # Decidir melhor ação baseada em scores
        return self.decide_best_action(available_actions, shared_memory)
    
    def decide_best_action(self, available_cells, shared_memory):
        """Decide melhor ação usando pesos específicos da estratégia"""
        if not available_cells:
            return None
        
        best_score = -float('inf')
        best_action = None
        
        for pos, predicted_type in available_cells:
            # Score base
            score = self.weights.get(predicted_type, 0.0)
            
            # Bônus se bandeira descoberta e caminho encurta
            if shared_memory.flag_position:
                curr_dist = abs(self.position[0] - shared_memory.flag_position[0]) + \
                           abs(self.position[1] - shared_memory.flag_position[1])
                new_dist = abs(pos[0] - shared_memory.flag_position[0]) + \
                          abs(pos[1] - shared_memory.flag_position[1])
                if new_dist < curr_dist:
                    score += 25.0
            
            # Bônus por exploração (bandeira ainda não descoberta)
            if not shared_memory.cell_knowledge[pos]['explored']:
                score += 5.0
            
            # Penalidade por proximidade a bombas
            for bx, by in shared_memory.bombs_found:
                dist = abs(pos[0] - bx) + abs(pos[1] - by)
                if dist == 1:
                    score -= 40.0
            
            # Aleatoriedade
            score += random.uniform(-0.1, 0.1)
            
            if score > best_score:
                best_score = score
                best_action = pos
        
        return best_action
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente e atualiza memória compartilhada"""
        if not self.alive or new_position is None:
            return "Agente inativo"
        
        old_pos = self.position
        self.position = new_position
        self.steps_taken += 1
        self.action_history.append(old_pos)
        self.path_cost += shared_memory.cell_knowledge[new_position]['cost']
        
        # Explorar célula
        x, y = new_position
        cell_content = env.get_cell(x, y)
        
        # Atualizar memória compartilhada
        log_msg = shared_memory.update_explored(new_position, cell_content, self.id, env)
        
        # Atualizar dados de treinamento próprios
        self.training_data['features'].append([x, y])
        self.training_data['labels'].append(cell_content)
        
        # Consequências
        if cell_content == 'T':
            if new_position in shared_memory.treasures_collected:
                self.treasures_collected += 1
                self.bombs_defused += 1
                log_msg += f" | Tesouros: {self.treasures_collected} [{self.ml_algorithm}]"
        elif cell_content == 'B':
            if self.bombs_defused > 0:
                self.bombs_defused -= 1
                log_msg += f" | BOMBA DESATIVADA! [{self.ml_algorithm}]"
                shared_memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += f" | AGENTE DESTRUÍDO [{self.ml_algorithm}]"
        elif cell_content == 'F':
            log_msg += f" | Custo: {self.path_cost:.2f} [{self.ml_algorithm}]"
        
        return log_msg
    
    def train_models(self, shared_memory=None, env=None):
        """Compatibilidade com GUI"""
        self.train_model(shared_memory, env)

# ============================================
# 6. SIMULAÇÃO DA ABORDAGEM C (GRUPOS ML)
# ============================================

class ApproachCSimulation:
    def __init__(self, num_agents=4, bomb_ratio=0.3, treasure_count=10, 
                 homogeneous=True, max_steps=500):
        self.env = EnvironmentC(bomb_ratio=bomb_ratio, treasure_count=treasure_count)
        # IMPORTANTE: Passar flag_position apenas para GUI (true_flag_position)
        self.shared_memory = SharedMemoryC(flag_position=self.env.flag_position)
        self.num_agents = num_agents
        self.homogeneous = homogeneous
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        self.metrics = {
            'flag_found': False,
            'treasures_found': 0,
            'total_treasures': treasure_count,
            'agents_alive': 0,
            'steps_taken': 0,
            'execution_time': 0,
            'success': False,
            'bomb_ratio': bomb_ratio,
            'min_path_cost': float('inf'),
            'avg_path_cost': 0,
            'explored_percentage': 0
        }
        self.setup_agents()
        
    def setup_agents(self):
        """Configura agentes"""
        base_weights = {
            'F': 100.0, 'T': 8.0, 'L': 2.0, 
            'B': -200.0, 'U': 3.0, 'E': -0.8
        }
        
        if self.homogeneous:
            for i in range(self.num_agents):
                agent = AgentC(agent_id=i, inference_weights=base_weights)
                self.agents.append(agent)
        else:
            # Perfis diferentes para exploração
            profiles = [
                {'F': 120.0, 'T': 6.0, 'L': 1.0, 'B': -180.0, 'U': 4.5, 'E': -0.6},  # Explorador
                {'F': 100.0, 'T': 10.0, 'L': 2.0, 'B': -220.0, 'U': 2.8, 'E': -1.0}, # Cauteloso
                {'F': 110.0, 'T': 7.0, 'L': 1.5, 'B': -200.0, 'U': 3.5, 'E': -0.7},  # Equilibrado
                {'F': 130.0, 'T': 5.0, 'L': 0.8, 'B': -190.0, 'U': 4.2, 'E': -0.5},  # Focado
                {'F': 100.0, 'T': 9.0, 'L': 2.5, 'B': -210.0, 'U': 3.0, 'E': -0.9}   # Coletor
            ]
            
            for i in range(self.num_agents):
                profile = profiles[i % len(profiles)]
                agent = AgentC(agent_id=i, inference_weights=profile)
                self.agents.append(agent)
    
    def run_simulation(self, verbose=False):
        """Executa simulação completa"""
        start_time = time.time()
        step = 0
        
        if verbose:
            self.logs.append(f"=== INÍCIO SIMULAÇÃO ABORDAGEM C ===")
            self.logs.append(f"Agentes: {self.num_agents} | Objetivo: Encontrar bandeira DESCONHECIDA")
            self.logs.append(f"Posição real da bandeira: {self.env.flag_position} (OCULTA dos agentes)")
            self.logs.append(f"Bombas: {self.metrics['bomb_ratio']*100}%")
            self.logs.append(f"Tipo: {'Homogêneo' if self.homogeneous else 'Heterogêneo'}")
        
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                next_pos = agent.choose_action(self.shared_memory, self.env)
                if next_pos:
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
                    if verbose and ("TESOURO" in log_msg or "BOMBA" in log_msg or "BANDEIRA" in log_msg):
                        self.logs.append(log_msg)
            
            if verbose and step % 50 == 0:
                self.logs.append(f"Passo {step}: Bandeira {'ENCONTRADA' if self.shared_memory.flag_found else 'não encontrada'}, {agents_alive} agentes vivos")
            
            # Verificar sucesso
            if self.shared_memory.flag_found:
                self.metrics['success'] = True
                if verbose:
                    self.logs.append(f"✅ SUCESSO! Bandeira encontrada no passo {step}!")
                break
        
        # Métricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['flag_found'] = self.shared_memory.flag_found
        self.metrics['treasures_found'] = len(self.shared_memory.treasures_collected)
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        self.metrics['explored_percentage'] = self.get_explored_percentage()
        
        alive_agents = [a for a in self.agents if a.alive]
        if alive_agents:
            path_costs = [a.path_cost for a in alive_agents]
            self.metrics['min_path_cost'] = min(path_costs)
            self.metrics['avg_path_cost'] = np.mean(path_costs)
        
        if verbose:
            self.logs.append(f"\n=== FIM DA SIMULAÇÃO ===")
            self.logs.append(f"Tempo: {self.metrics['execution_time']:.2f}s")
            self.logs.append(f"Passos: {self.metrics['steps_taken']}")
            self.logs.append(f"Bandeira: {'Encontrada' if self.metrics['flag_found'] else 'Não encontrada'}")
            self.logs.append(f"Tesouros: {self.metrics['treasures_found']}/{self.env.treasure_count}")
            self.logs.append(f"Agentes vivos: {self.metrics['agents_alive']}")
            self.logs.append(f"Custo mínimo: {self.metrics['min_path_cost']:.2f}")
            self.logs.append(f"Sucesso: {'SIM' if self.metrics['success'] else 'NÃO'}")
        
        return self.metrics
    
    def get_explored_percentage(self):
        """Calcula percentagem de células exploradas"""
        explored_count = len(self.shared_memory.explored)
        total_cells = self.env.size * self.env.size
        return (explored_count / total_cells) * 100
    
    def print_logs(self):
        """Exibe logs da simulação"""
        for log in self.logs:
            print(log)

# ============================================
# 7. BASELINE ML - MÚLTIPLOS AGENTES HETEROGÊNEOS
# ============================================

class BaselineC_ML:
    """Baseline C: Agentes ML heterogêneos com colaboração"""
    def __init__(self, num_agents=4, bomb_ratio=0.3, treasure_count=10, max_steps=500):
        self.env = EnvironmentC(bomb_ratio=bomb_ratio, treasure_count=treasure_count)
        # IMPORTANTE: Bandeira desconhecida para baseline também
        self.shared_memory = SharedMemoryC(flag_position=self.env.flag_position)
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        
        self.metrics = {
            'flag_found': False,
            'execution_time': 0,
            'steps_taken': 0,
            'success': False,
            'path_length': 0,
            'treasures_found': 0,
            'total_treasures': treasure_count,
            'agents_alive': num_agents,
            'min_path_cost': float('inf'),
            'avg_path_cost': 0
        }
        
        self.setup_agents()
    
    def setup_agents(self):
        """Criar agentes ML heterogêneos"""
        # Lista de algoritmos disponíveis
        algorithms = ['KNN', 'NaiveBayes', 'DecisionTree']
        
        for i in range(self.num_agents):
            # Distribuir algoritmos ciclicamente
            algo = algorithms[i % len(algorithms)]
            agent = AgentML_Baseline(agent_id=i, ml_algorithm=algo, start_pos=(0, 0))
            self.agents.append(agent)
            print(f"Baseline ML - Agente {i}: {algo} ({agent.strategy})")
    
    def run(self):
        """Executar simulação com agentes ML heterogêneos"""
        start_time = time.time()
        step = 0
        
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            # Cada agente age com seu modelo ML
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                next_pos = agent.choose_action(self.shared_memory, self.env)
                
                if next_pos:
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
            
            # Verificar se bandeira foi encontrada
            if self.shared_memory.flag_found:
                self.metrics['success'] = True
                self.metrics['flag_found'] = True
                break
        
        # Métricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        self.metrics['treasures_found'] = len(self.shared_memory.treasures_collected)
        
        # Caminho do agente que encontrou
        alive_agents = [a for a in self.agents if a.alive]
        if alive_agents:
            path_costs = [a.path_cost for a in alive_agents]
            self.metrics['min_path_cost'] = min(path_costs)
            self.metrics['avg_path_cost'] = np.mean(path_costs)
            self.metrics['path_length'] = min([a.steps_taken for a in alive_agents if a.steps_taken > 0])
        
        return self.metrics

# Alias para manter compatibilidade com GUI
BaselineC_AStar = BaselineC_ML

# ============================================
# 8. FUNÇÕES DE ANÁLISE
# ============================================

def run_multiple_simulations_c(num_simulations=5, num_agents=4, homogeneous=True):
    """Executa múltiplas simulações para estatísticas"""
    results = []
    
    for i in range(num_simulations):
        sim = ApproachCSimulation(
            num_agents=num_agents,
            bomb_ratio=0.3,
            treasure_count=10,
            homogeneous=homogeneous,
            max_steps=300
        )
        
        metrics = sim.run_simulation(verbose=False)
        results.append(metrics)
    
    # Calcular médias
    avg_flag_found = np.mean([1 if r['flag_found'] else 0 for r in results])
    avg_time = np.mean([r['execution_time'] for r in results])
    success_rate = np.mean([1 if r['success'] else 0 for r in results])
    avg_survivors = np.mean([r['agents_alive'] for r in results])
    avg_path_cost = np.mean([r['min_path_cost'] for r in results if r['min_path_cost'] != float('inf')])
    
    return {
        'type': 'Homogêneo' if homogeneous else 'Heterogêneo',
        'num_agents': num_agents,
        'avg_flag_found': avg_flag_found,
        'avg_time': avg_time,
        'success_rate': success_rate,
        'avg_survivors': avg_survivors,
        'avg_path_cost': avg_path_cost,
        'results': results
    }

def compare_approaches_c():
    """Compara abordagens homogênea, heterogênea e baseline ML"""
    print("Comparando Abordagem C...")
    
    results = []
    agent_counts = [3, 4, 6, 8]
    
    for num_agents in agent_counts:
        print(f"\nTestando com {num_agents} agentes...")
        
        # Homogêneo
        homo_results = run_multiple_simulations_c(
            num_simulations=5, num_agents=num_agents, homogeneous=True
        )
        
        # Heterogêneo
        hetero_results = run_multiple_simulations_c(
            num_simulations=5, num_agents=num_agents, homogeneous=False
        )
        
        # Baseline ML heterogêneo
        ml_baseline_results = []
        for _ in range(5):
            baseline = BaselineC_ML(num_agents=num_agents)
            baseline_metrics = baseline.run()
            ml_baseline_results.append(baseline_metrics)
        
        ml_success_rate = np.mean([1 if r['success'] else 0 for r in ml_baseline_results])
        ml_avg_time = np.mean([r['execution_time'] for r in ml_baseline_results])
        
        results.append({
            'num_agents': num_agents,
            'homogeneous': homo_results,
            'heterogeneous': hetero_results,
            'ml_baseline': {
                'success_rate': ml_success_rate,
                'avg_time': ml_avg_time,
                'results': ml_baseline_results
            }
        })
        
        print(f"  Homogêneo: {homo_results['success_rate']:.0%} sucesso")
        print(f"  Heterogêneo: {hetero_results['success_rate']:.0%} sucesso")
        print(f"  Baseline ML: {ml_success_rate:.0%} sucesso")
    
    return results

# ============================================
# 9. EXECUÇÃO PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("PROJETO IA - ABORDAGEM C: BUSCA DA BANDEIRA (DESCONHECIDA)")
    print("="*60)
    
    # Teste rápido
    print("\n1. Teste rápido - 4 agentes heterogêneos:")
    sim = ApproachCSimulation(num_agents=4, homogeneous=False, max_steps=300)
    metrics = sim.run_simulation(verbose=True)
    sim.print_logs()
    
    # Teste Baseline ML
    print("\n\n2. Teste Baseline ML - 4 agentes heterogêneos:")
    baseline = BaselineC_ML(num_agents=4)
    baseline_metrics = baseline.run()
    print(f"Baseline ML: {'Sucesso' if baseline_metrics['success'] else 'Falha'}")
    print(f"Passos: {baseline_metrics['steps_taken']}")
    print(f"Tempo: {baseline_metrics['execution_time']:.2f}s")
    
    # Comparação completa
    print("\n\n3. Comparação completa...")
    results = compare_approaches_c()
    
    print("\n\n4. ANÁLISE DOS RESULTADOS:")
    print("-"*50)
    
    best_homo = max(results, key=lambda x: x['homogeneous']['success_rate'])
    best_hetero = max(results, key=lambda x: x['heterogeneous']['success_rate'])
    
    print(f"\nMelhor configuração Homogênea: {best_homo['num_agents']} agentes")
    print(f"  Taxa de sucesso: {best_homo['homogeneous']['success_rate']:.0%}")
    
    print(f"\nMelhor configuração Heterogênea: {best_hetero['num_agents']} agentes")
    print(f"  Taxa de sucesso: {best_hetero['heterogeneous']['success_rate']:.0%}")
    
    print(f"\nBaseline ML Heterogêneo:")
    print(f"  Usa KNN, Naive Bayes e Decision Tree com colaboração")
    print(f"  Cada agente tem estratégia única baseada em seu algoritmo")