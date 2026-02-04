# abordagem_c.py
# Abordagem C: Encontrar a bandeira com otimizaÃ§Ã£o de caminho
# Objetivo: Localizar objetivo especÃ­fico (bandeira) minimizando custo, risco e passos

import numpy as np
import random
import time
import warnings
from collections import defaultdict, deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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
        self.flag_position = None  # PosiÃ§Ã£o da bandeira (objetivo)
        self.generate_environment()
        
    def generate_environment(self):
        """Gera ambiente com tesouros, bombas e uma bandeira"""
        total_cells = self.size * self.size
        all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        # Garantir que tesouros nÃ£o sejam mais que 20% do ambiente
        max_treasures = int(total_cells * 0.2)
        self.treasure_count = min(self.treasure_count, max_treasures)
        
        # Escolher posiÃ§Ãµes para tesouros
        treasure_positions = random.sample(all_positions, self.treasure_count)
        
        # Escolher posiÃ§Ã£o para bandeira (nÃ£o em tesouro)
        remaining_positions = [pos for pos in all_positions if pos not in treasure_positions]
        self.flag_position = random.choice(remaining_positions)
        remaining_positions.remove(self.flag_position)
        
        # Calcular nÃºmero de bombas
        bomb_count = int((total_cells - self.treasure_count - 1) * self.bomb_ratio)  # -1 para bandeira
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
        """Remove tesouro apÃ³s ser coletado"""
        x, y = position
        self.grid[x, y] = 'L'
        
    def get_cell(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x, y]
        return None
    
    def get_neighbors(self, x, y):
        """Retorna vizinhas vÃ¡lidas (apenas horizontal/vertical)"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Apenas 4 direÃ§Ãµes
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors

# ============================================
# 2. MEMÃ“RIA COMPARTILHADA PARA ABORDAGEM C
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
        self.flag_position = flag_position  # BANDEIRA JÃ CONHECIDA DESDE O INÃCIO
        
        # Custos estimados de deslocaÃ§Ã£o (para otimizaÃ§Ã£o de caminho)
        self.movement_costs = {}
        
        # Inicializar conhecimento
        for i in range(env_size):
            for j in range(env_size):
                self.cell_knowledge[(i, j)] = {
                    'type': 'U',  # U = Unknown
                    'explored': False,
                    'safe': True,  # Assume-se seguro atÃ© descobrir bomba
                    'cost': 1.0,   # Custo base de movimentaÃ§Ã£o
                    'risk': 0.0    # Risco estimado
                }
                self.movement_costs[(i, j)] = float('inf')
        
        # PosiÃ§Ã£o inicial (0,0) tem custo 0
        self.movement_costs[(0, 0)] = 0
        
        # MARCAR POSIÃ‡ÃƒO DA BANDEIRA COMO CONHECIDA (mas nÃ£o explorada)
        if self.flag_position:
            self.cell_knowledge[self.flag_position]['type'] = 'F'  # Tipo conhecido desde o inÃ­cio
    
    def update_explored(self, position, content, agent_id, env):
        """Atualiza memÃ³ria com nova exploraÃ§Ã£o"""
        x, y = position
        self.explored.add(position)
        self.agent_positions[agent_id] = position
        
        # Atualizar conhecimento da cÃ©lula
        self.cell_knowledge[position]['explored'] = True
        
        log_msg = f"Agente {agent_id}: {content} em {position}"
        
        if content == 'T':
            # Primeiro agente a encontrar o tesouro
            if position not in self.treasures_collected:
                self.treasures_found.add(position)
                self.treasures_collected.add(position)
                # Marcar tipo como tesouro ANTES de resetar
                self.cell_knowledge[position]['type'] = 'T'
                env.reset_treasure(position)  # Transforma tesouro em 'L' no ambiente
                log_msg += " (TESOURO COLETADO!)"
            else:
                # Tesouro jÃ¡ foi coletado por outro agente, agora Ã© cÃ©lula livre
                self.cell_knowledge[position]['type'] = 'L'
                log_msg += " (LIVRE - Tesouro jÃ¡ coletado)"
        elif content == 'B':
            self.cell_knowledge[position]['type'] = content
            self.bombs_found.add(position)
            self.cell_knowledge[position]['safe'] = False
            self.cell_knowledge[position]['risk'] = 1.0
            self.cell_knowledge[position]['cost'] = float('inf')
            log_msg += " (BOMBA)"
        elif content == 'F':
            self.cell_knowledge[position]['type'] = content
            self.flag_found = True
            self.flag_position = position
            log_msg += " (BANDEIRA ENCONTRADA! OBJETIVO ALCANÃ‡ADO!)"
        else:
            # CÃ©lula livre
            self.cell_knowledge[position]['type'] = content
            self.cell_knowledge[position]['safe'] = True
            
        # Atualizar custos das cÃ©lulas vizinhas
        self.update_neighbor_costs(position, env)
            
        return log_msg
    
    def update_neighbor_costs(self, position, env):
        """Atualizar custos e riscos das cÃ©lulas vizinhas"""
        x, y = position
        neighbors = env.get_neighbors(x, y)
        
        for neighbor in neighbors:
            if not self.cell_knowledge[neighbor]['explored']:
                # Aumentar risco se prÃ³ximo de bomba
                if position in self.bombs_found:
                    self.cell_knowledge[neighbor]['risk'] += 0.3
                    self.cell_knowledge[neighbor]['cost'] += 0.5
                else:
                    # Reduzir risco se prÃ³ximo de cÃ©lula segura
                    self.cell_knowledge[neighbor]['risk'] = max(0, self.cell_knowledge[neighbor]['risk'] - 0.1)
    
    def is_safe_cell(self, position):
        """Verifica se cÃ©lula Ã© segura para visitar"""
        return self.cell_knowledge[position]['safe']
    
    def estimate_distance_to_flag(self, position):
        """Estima distÃ¢ncia atÃ© a bandeira (heurÃ­stica)"""
        if self.flag_position:
            # Se jÃ¡ encontramos a bandeira, usar distÃ¢ncia real
            return abs(position[0] - self.flag_position[0]) + abs(position[1] - self.flag_position[1])
        else:
            # HeurÃ­stica: distÃ¢ncia atÃ© o centro do grid
            center = (self.env_size // 2, self.env_size // 2)
            return abs(position[0] - center[0]) + abs(position[1] - center[1])
    
    def get_best_path_neighbor(self, position, env):
        """Retorna melhor vizinha considerando custo e direÃ§Ã£o ao objetivo"""
        x, y = position
        neighbors = env.get_neighbors(x, y)
        
        best_neighbor = None
        best_score = float('inf')
        
        for neighbor in neighbors:
            if not self.cell_knowledge[neighbor]['explored'] and self.cell_knowledge[neighbor]['safe']:
                # Score = custo + risco + distÃ¢ncia estimada ao objetivo
                cost = self.cell_knowledge[neighbor]['cost']
                risk = self.cell_knowledge[neighbor]['risk']
                distance = self.estimate_distance_to_flag(neighbor)
                
                score = cost + (risk * 2.0) + (distance * 0.5)
                
                if score < best_score:
                    best_score = score
                    best_neighbor = neighbor
        
        return best_neighbor

# ============================================
# 3. AGENTE PARA ABORDAGEM C
# ============================================

class AgentC:
    def __init__(self, agent_id, start_pos=(0, 0), inference_weights=None):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.treasures_collected = 0
        self.bombs_defused = 0
        self.steps_taken = 0
        self.path_cost = 0  # Custo total do caminho percorrido
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
        
        # Motor de inferÃªncia otimizado para busca de objetivo
        self.inference_engine = InferenceEngineC(inference_weights)
        
        # HistÃ³rico de aÃ§Ãµes
        self.action_history = deque(maxlen=50)
        
        # MemÃ³ria individual do agente (para baseline)
        self.memory = SharedMemoryC()
        
        # Inicializar com algum conhecimento bÃ¡sico
        self.initialize_basic_knowledge()
    
    def initialize_basic_knowledge(self):
        """Inicializa conhecimento bÃ¡sico"""
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
        """PrevÃª o tipo de cÃ©lula usando modelos ou heurÃ­stica"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        x, y = cell_position
        
        # Se cÃ©lula jÃ¡ foi explorada, retorna conteÃºdo conhecido
        if cell_position in memory.explored:
            if cell_position in memory.treasures_collected:
                return 'L'
            elif cell_position in memory.bombs_found:
                return 'B'
            elif memory.flag_found and cell_position == memory.flag_position:
                return 'F'
            else:
                return 'L'
        
        # Se modelos estÃ£o treinados, usar previsÃ£o
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
        
        # HeurÃ­stica: cÃ©lulas mais distantes do centro tÃªm maior risco
        distance_to_center = np.sqrt((x - 5)**2 + (y - 5)**2)
        if distance_to_center < 3:
            return random.choice(['L', 'L', 'L'])
        else:
            return random.choice(['L', 'L', 'B', 'L'])
    
    def choose_action(self, shared_memory, env):
        """Escolhe prÃ³xima aÃ§Ã£o otimizando caminho atÃ© a bandeira"""
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
        
        # Filtrar apenas cÃ©lulas seguras
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
        
        # Usar motor de inferÃªncia para decidir
        next_pos = self.inference_engine.decide_action(
            available_actions, memory, env, self
        )
        
        return next_pos
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posiÃ§Ã£o"""
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
        
        # Explorar nova cÃ©lula
        x, y = new_position
        cell_content = env.get_cell(x, y)
        
        # Atualizar memÃ³ria compartilhada
        if shared_memory is None:
            log_msg = memory.update_explored(new_position, cell_content, self.id, env)
        else:
            log_msg = shared_memory.update_explored(new_position, cell_content, self.id, env)
        
        # Para baseline, atualizar tambÃ©m memÃ³ria individual
        if getattr(self, 'is_baseline', False):
            memory.update_explored(new_position, cell_content, self.id, env)
        
        # Atualizar dados de treinamento
        self.training_data['features'].append([x, y])
        self.training_data['labels'].append(cell_content)
        
        # ConsequÃªncias da aÃ§Ã£o
        if cell_content == 'T':
            # SÃ³ incrementa se realmente coletou (primeiro a chegar)
            if new_position in (shared_memory.treasures_collected if shared_memory else memory.treasures_collected):
                self.treasures_collected += 1
                self.bombs_defused += 1
                log_msg += f" | Tesouros: {self.treasures_collected}"
        elif cell_content == 'L':
            # Verificar se era um tesouro que jÃ¡ foi coletado
            if new_position in (shared_memory.treasures_collected if shared_memory else memory.treasures_collected):
                # Apenas para informaÃ§Ã£o, nÃ£o coleta novamente
                pass
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
                log_msg += " | AGENTE DESTRUÃDO"
        elif cell_content == 'F':
            log_msg += f" | CUSTO TOTAL DO CAMINHO: {self.path_cost:.2f}"
        
        return log_msg

# ============================================
# 4. MOTOR DE INFERÃŠNCIA PARA ABORDAGEM C
# ============================================

class InferenceEngineC:
    def __init__(self, weights=None):
        # Pesos otimizados para busca de objetivo
        self.weights = weights or {
            'F': 100.0,   # Bandeira - prioridade mÃ¡xima
            'T': 8.0,     # Tesouro - importante para desativar bombas
            'L': 2.0,     # Livre
            'B': -200.0,  # Bomba - evitar completamente
            'U': 1.0,     # Desconhecido
            'E': -0.8     # Explorado - evitar revisitar
        }
        
        # Pesos para fatores de otimizaÃ§Ã£o
        self.cost_weight = 0.5      # Peso do custo de movimento
        self.risk_weight = 2.0      # Peso do risco
        self.distance_weight = 1.5  # Peso da distÃ¢ncia ao objetivo
    
    def calculate_score(self, cell_type, position, shared_memory, agent):
        """Calcula pontuaÃ§Ã£o otimizada para busca de objetivo"""
        base_score = self.weights.get(cell_type, 0.0)
        
        # Bonus se a cÃ©lula estÃ¡ na direÃ§Ã£o da bandeira
        if shared_memory.flag_position:
            current_dist = shared_memory.estimate_distance_to_flag(agent.position)
            new_dist = shared_memory.estimate_distance_to_flag(position)
            if new_dist < current_dist:
                base_score += 15.0  # Grande bonus por aproximar do objetivo
        
        # Penalidade por custo e risco
        cost = shared_memory.cell_knowledge[position]['cost']
        risk = shared_memory.cell_knowledge[position]['risk']
        
        base_score -= (cost * self.cost_weight)
        base_score -= (risk * self.risk_weight)
        
        # Penalidade por distÃ¢ncia ao objetivo
        distance = shared_memory.estimate_distance_to_flag(position)
        base_score -= (distance * self.distance_weight)
        
        # Penalidade por estar perto de bomba
        x, y = position
        for bx, by in shared_memory.bombs_found:
            bomb_distance = abs(bx - x) + abs(by - y)
            if bomb_distance == 1:
                base_score -= 30.0
        
        # Bonus por exploraÃ§Ã£o de Ã¡rea nova
        if not shared_memory.cell_knowledge[position]['explored']:
            base_score += 3.0
        
        return base_score
    
    def decide_action(self, available_cells, shared_memory, env, agent):
        """Decide aÃ§Ã£o otimizando caminho atÃ© objetivo"""
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
# 5. SIMULAÃ‡ÃƒO DA ABORDAGEM C
# ============================================

class ApproachCSimulation:
    def __init__(self, num_agents=4, bomb_ratio=0.3, treasure_count=10, 
                 homogeneous=True, max_steps=500):
        self.env = EnvironmentC(bomb_ratio=bomb_ratio, treasure_count=treasure_count)
        self.shared_memory = SharedMemoryC(flag_position=self.env.flag_position)  # Passar posiÃ§Ã£o da bandeira
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
        # Pesos base otimizados para busca
        base_weights = {
            'F': 100.0, 'T': 8.0, 'L': 2.0, 
            'B': -200.0, 'U': 1.0, 'E': -0.8
        }
        
        if self.homogeneous:
            for i in range(self.num_agents):
                agent = AgentC(agent_id=i, inference_weights=base_weights)
                self.agents.append(agent)
        else:
            # Perfis diferentes enfatizando estratÃ©gias distintas
            profiles = [
                {'F': 120.0, 'T': 6.0, 'L': 1.0, 'B': -180.0, 'U': 1.5, 'E': -0.6},  # Agressivo
                {'F': 100.0, 'T': 10.0, 'L': 2.0, 'B': -220.0, 'U': 0.8, 'E': -1.0}, # Cauteloso
                {'F': 110.0, 'T': 7.0, 'L': 1.5, 'B': -200.0, 'U': 1.0, 'E': -0.7},  # Equilibrado
                {'F': 130.0, 'T': 5.0, 'L': 0.8, 'B': -190.0, 'U': 1.2, 'E': -0.5},  # Focado
                {'F': 100.0, 'T': 9.0, 'L': 2.5, 'B': -210.0, 'U': 0.9, 'E': -0.9}   # Explorador
            ]
            
            for i in range(self.num_agents):
                profile = profiles[i % len(profiles)]
                agent = AgentC(agent_id=i, inference_weights=profile)
                self.agents.append(agent)
    
    def run_simulation(self, verbose=False):
        """Executa simulaÃ§Ã£o completa"""
        start_time = time.time()
        step = 0
        
        if verbose:
            self.logs.append(f"=== INÃCIO SIMULAÃ‡ÃƒO ABORDAGEM C ===")
            self.logs.append(f"Agentes: {self.num_agents} | Objetivo: Encontrar bandeira")
            self.logs.append(f"PosiÃ§Ã£o da bandeira: {self.env.flag_position}")
            self.logs.append(f"Bombas: {self.metrics['bomb_ratio']*100}%")
            self.logs.append(f"Tipo: {'HomogÃªneo' if self.homogeneous else 'HeterogÃªneo'}")
        
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            step_logs = []
            
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # Escolher e executar aÃ§Ã£o
                next_pos = agent.choose_action(self.shared_memory, self.env)
                if next_pos:
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
                    if verbose and ("TESOURO" in log_msg or "BOMBA" in log_msg or "BANDEIRA" in log_msg):
                        step_logs.append(log_msg)
            
            # Atualizar mÃ©tricas
            if verbose and step % 50 == 0:
                self.logs.append(f"Passo {step}: Bandeira {'ENCONTRADA' if self.shared_memory.flag_found else 'nÃ£o encontrada'}, {agents_alive} agentes vivos")
            
            # Verificar critÃ©rio de sucesso: bandeira encontrada
            if self.shared_memory.flag_found:
                self.metrics['success'] = True
                if verbose:
                    self.logs.append(f"âœ… SUCESSO! Bandeira encontrada no passo {step}!")
                break
        
        # Calcular mÃ©tricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['flag_found'] = self.shared_memory.flag_found
        self.metrics['treasures_found'] = len(self.shared_memory.treasures_collected)
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        self.metrics['explored_percentage'] = self.get_explored_percentage()
        
        # Calcular custos de caminho
        alive_agents = [a for a in self.agents if a.alive]
        if alive_agents:
            path_costs = [a.path_cost for a in alive_agents]
            self.metrics['min_path_cost'] = min(path_costs)
            self.metrics['avg_path_cost'] = np.mean(path_costs)
        
        if verbose:
            self.logs.append(f"\n=== FIM DA SIMULAÃ‡ÃƒO ===")
            self.logs.append(f"Tempo: {self.metrics['execution_time']:.2f}s")
            self.logs.append(f"Passos: {self.metrics['steps_taken']}")
            self.logs.append(f"Bandeira: {'Encontrada' if self.metrics['flag_found'] else 'NÃ£o encontrada'}")
            self.logs.append(f"Tesouros: {self.metrics['treasures_found']}/{self.env.treasure_count}")
            self.logs.append(f"Agentes vivos: {self.metrics['agents_alive']}")
            self.logs.append(f"Custo mÃ­nimo: {self.metrics['min_path_cost']:.2f}")
            self.logs.append(f"Sucesso: {'SIM' if self.metrics['success'] else 'NÃƒO'}")
        
        return self.metrics
    
    def get_explored_percentage(self):
        """Calcula percentagem de cÃ©lulas exploradas"""
        explored_count = len(self.shared_memory.explored)
        total_cells = self.env.size * self.env.size
        return (explored_count / total_cells) * 100
    
    def print_logs(self):
        """Exibe logs da simulaÃ§Ã£o"""
        for log in self.logs:
            print(log)

# ============================================
# 6. BASELINE: A* COLABORATIVO PARA ABORDAGEM C
# ============================================

class AgentAStar:
    """Agente que usa A* para encontrar a bandeira"""
    def __init__(self, agent_id, start_pos=(0, 0)):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.steps_taken = 0
        self.treasures_collected = 0
        self.bombs_defused = 0
        self.path = []  # Caminho planejado
        self.path_index = 0  # Ãndice atual no caminho
        self.flag_found = False
        
    def plan_path_astar(self, goal, env, shared_memory):
        """Planeja caminho usando A* evitando bombas conhecidas"""
        import heapq
        
        start = self.position
        
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
            
            for neighbor in env.get_neighbors(*current):
                # Evitar APENAS bombas JÁ CONHECIDAS (exploradas anteriormente)
                if neighbor in shared_memory.bombs_found:
                    continue
                
                # NÃO olhar para o grid diretamente durante planejamento
                # Isso seria trapaça! O agente só descobre bombas ao explorá-las
                # Assumir que células não exploradas são seguras para planejamento
                
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
        
        # Reconstruir caminho
        if goal in came_from:
            path = []
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        return []  # Sem caminho
    
    def choose_action(self, shared_memory, env):
        """Escolhe prÃ³xima aÃ§Ã£o baseada no caminho A*"""
        if not self.alive:
            return None
        
        # Se bandeira jÃ¡ foi encontrada por outro agente, parar
        if shared_memory.flag_found:
            return None
        
        # Se nÃ£o tem caminho ou caminho foi invalidado, replanejear
        if not self.path or self.path_index >= len(self.path):
            goal = shared_memory.flag_position
            if goal:
                self.path = self.plan_path_astar(goal, env, shared_memory)
                self.path_index = 0
        
        # Se ainda nÃ£o tem caminho vÃ¡lido, ficar parado
        if not self.path or self.path_index >= len(self.path):
            return None
        
        # Retornar prÃ³ximo passo no caminho
        next_pos = self.path[self.path_index]
        self.path_index += 1
        return next_pos
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posiÃ§Ã£o"""
        if not self.alive or new_position is None:
            return "Agente inativo"
        
        old_pos = self.position
        self.position = new_position
        self.steps_taken += 1
        
        # Explorar nova cÃ©lula
        x, y = new_position
        cell_content = env.get_cell(x, y)
        
        # Atualizar memÃ³ria compartilhada
        log_msg = shared_memory.update_explored(new_position, cell_content, self.id, env)
        
        # ConsequÃªncias da aÃ§Ã£o
        if cell_content == 'T':
            self.treasures_collected += 1
            self.bombs_defused += 1
            log_msg += f" | Tesouros: {self.treasures_collected}"
        elif cell_content == 'B':
            if self.bombs_defused > 0:
                self.bombs_defused -= 1
                log_msg += " | BOMBA DESATIVADA!"
                shared_memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " | AGENTE DESTRUÃDO"
                # Invalidar caminho
                self.path = []
        elif cell_content == 'F':
            self.flag_found = True
            log_msg += " | ðŸš© BANDEIRA ENCONTRADA!"
        
        return log_msg
    
    def train_models(self, shared_memory=None, env=None):
        """Compatibilidade com GUI - nÃ£o usado em A*"""
        pass


class BaselineC_AStar:
    """Baseline C: MÃºltiplos agentes usando A* colaborativo"""
    def __init__(self, num_agents=4, bomb_ratio=0.3, treasure_count=10, max_steps=500):
        self.env = EnvironmentC(bomb_ratio=bomb_ratio, treasure_count=treasure_count)
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
            'agents_alive': num_agents
        }
        
        # Criar agentes
        self.setup_agents()
    
    def setup_agents(self):
        """Criar mÃºltiplos agentes A*"""
        for i in range(self.num_agents):
            agent = AgentAStar(agent_id=i, start_pos=(0, 0))
            self.agents.append(agent)
    
    def run(self):
        """Executar simulaÃ§Ã£o com mÃºltiplos agentes A*"""
        start_time = time.time()
        step = 0
        
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            # Cada agente executa sua aÃ§Ã£o
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # Escolher prÃ³xima posiÃ§Ã£o usando A*
                next_pos = agent.choose_action(self.shared_memory, self.env)
                
                if next_pos:
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
            
            # Verificar se bandeira foi encontrada
            if self.shared_memory.flag_found:
                self.metrics['success'] = True
                self.metrics['flag_found'] = True
                break
        
        # Calcular mÃ©tricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        self.metrics['treasures_found'] = len(self.shared_memory.treasures_collected)
        
        # Caminho do agente que encontrou (se houver)
        for agent in self.agents:
            if agent.flag_found:
                self.metrics['path_length'] = agent.steps_taken
                break
        
        return self.metrics

# ============================================
# 7. FUNÃ‡Ã•ES DE ANÃLISE E COMPARAÃ‡ÃƒO
# ============================================

def run_multiple_simulations_c(num_simulations=5, num_agents=4, homogeneous=True):
    """Executa mÃºltiplas simulaÃ§Ãµes para estatÃ­sticas"""
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
    
    # Calcular mÃ©dias
    avg_flag_found = np.mean([1 if r['flag_found'] else 0 for r in results])
    avg_time = np.mean([r['execution_time'] for r in results])
    success_rate = np.mean([1 if r['success'] else 0 for r in results])
    avg_survivors = np.mean([r['agents_alive'] for r in results])
    avg_path_cost = np.mean([r['min_path_cost'] for r in results if r['min_path_cost'] != float('inf')])
    
    return {
        'type': 'HomogÃªneo' if homogeneous else 'HeterogÃªneo',
        'num_agents': num_agents,
        'avg_flag_found': avg_flag_found,
        'avg_time': avg_time,
        'success_rate': success_rate,
        'avg_survivors': avg_survivors,
        'avg_path_cost': avg_path_cost,
        'results': results
    }

def compare_approaches_c():
    """Compara abordagens homogÃªnea, heterogÃªnea e baseline A*"""
    print("Comparando Abordagem C...")
    
    results = []
    agent_counts = [3, 4, 6, 8]
    
    for num_agents in agent_counts:
        print(f"\nTestando com {num_agents} agentes...")
        
        # HomogÃªneo
        homo_results = run_multiple_simulations_c(
            num_simulations=5, num_agents=num_agents, homogeneous=True
        )
        
        # HeterogÃªneo
        hetero_results = run_multiple_simulations_c(
            num_simulations=5, num_agents=num_agents, homogeneous=False
        )
        
        # Baseline A*
        astar_results = []
        for _ in range(5):
            astar = BaselineC_AStar()
            astar_metrics = astar.run()
            astar_results.append(astar_metrics)
        
        astar_success_rate = np.mean([1 if r['success'] else 0 for r in astar_results])
        astar_avg_time = np.mean([r['execution_time'] for r in astar_results])
        
        results.append({
            'num_agents': num_agents,
            'homogeneous': homo_results,
            'heterogeneous': hetero_results,
            'astar': {
                'success_rate': astar_success_rate,
                'avg_time': astar_avg_time,
                'results': astar_results
            }
        })
        
        print(f"  HomogÃªneo: {homo_results['success_rate']:.0%} sucesso")
        print(f"  HeterogÃªneo: {hetero_results['success_rate']:.0%} sucesso")
        print(f"  A*: {astar_success_rate:.0%} sucesso")
    
    return results

# ============================================
# 8. EXECUÃ‡ÃƒO PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("PROJETO IA - ABORDAGEM C: BUSCA DA BANDEIRA")
    print("="*60)
    
    # Teste rÃ¡pido
    print("\n1. Teste rÃ¡pido - 4 agentes heterogÃªneos:")
    sim = ApproachCSimulation(num_agents=4, homogeneous=False, max_steps=300)
    metrics = sim.run_simulation(verbose=True)
    sim.print_logs()
    
    # ComparaÃ§Ã£o completa
    print("\n\n2. ComparaÃ§Ã£o completa...")
    results = compare_approaches_c()
    
    print("\n\n3. ANÃLISE DOS RESULTADOS:")
    print("-"*50)
    
    best_homo = max(results, key=lambda x: x['homogeneous']['success_rate'])
    best_hetero = max(results, key=lambda x: x['heterogeneous']['success_rate'])
    
    print(f"\nMelhor configuraÃ§Ã£o HomogÃªnea: {best_homo['num_agents']} agentes")
    print(f"  Taxa de sucesso: {best_homo['homogeneous']['success_rate']:.0%}")
    
    print(f"\nMelhor configuraÃ§Ã£o HeterogÃªnea: {best_hetero['num_agents']} agentes")
    print(f"  Taxa de sucesso: {best_hetero['heterogeneous']['success_rate']:.0%}")