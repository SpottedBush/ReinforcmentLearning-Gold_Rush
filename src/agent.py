import numpy as np
from .terrain import Terrain
from .local_terrain import LocalTerrain

class Agent:
    def __init__(self,
                 terrain: Terrain,
                 trials: int,
                 class_type: str = 'uniform',
                 local_terrain: LocalTerrain = None,
                 epsilon_value: float = 0.5,
                 min_trials: int = 1,
                 softmax_temp: float = 1.0,
                 softmax_pure_exp_factor: float = 3.0):
        # Initialise l’agent avec le terrain et le nombre d’essais
        self.terrain = terrain
        self.trials = trials
        self.local_terrain = local_terrain if local_terrain else LocalTerrain(terrain.size_x, terrain.size_y) # Terrain that the Agent knows

        if class_type in ['uniform', 'epsilon_greedy', 'softmax', 'ucb']:
            self.class_type = class_type
        else:
            print('Invalid class type. Defaulting to uniform.')
            self.class_type = 'uniform'

        # epsilon-greedy Hyper-paramt
        self.epsilon_value = epsilon_value
        self.min_trials = min_trials 
        
        # softmax Hyper-paramters
        self.softmax_temp = softmax_temp
        self.softmax_t0 = terrain.size_x * terrain.size_y * softmax_pure_exp_factor
        self.softmax_iter = 0

    def select_position(self):
        # Sélectionne une position (x, y) sur le terrain
        coords = 0, 0
        if self.class_type == 'epsilon_greedy':
            coords = self._select_position_epsilon_greedy()
        elif self.class_type == 'softmax':
            coords = self._select_position_softmax()
        elif self.class_type == 'ucb':
            coords = self._select_position_ucb()
        else:
            coords = self._select_position_uniform()
        return coords
    

    def _select_position_uniform(self):
        return np.random.randint(0, self.terrain.size_x), np.random.randint(0, self.terrain.size_y)

    def _select_position_epsilon_greedy(self):
        if np.random.rand() < self.epsilon_value:
            not_yet_explored_values = [(i, j) for i in range(self.terrain.size_x) for j in range(self.terrain.size_y)
                                    if self.local_terrain[i, j]['trials'] <= self.min_trials]
            if not_yet_explored_values != []:
                return not_yet_explored_values[np.random.choice(range(len(not_yet_explored_values)))]
            else:
                return max(((i, j) for i in range(self.terrain.size_x) for j in range(self.terrain.size_y)),
                    key=lambda coord: self.local_terrain[coord]['mean'])

        else:
            return max(((i, j) for i in range(self.terrain.size_x) for j in range(self.terrain.size_y)),
                    key=lambda coord: self.local_terrain[coord]['mean'])

    def _select_position_softmax(self):
        if self.softmax_iter < self.softmax_t0:
            x_res = (self.softmax_iter // self.terrain.size_y) % self.terrain.size_x
            y_res = self.softmax_iter % self.terrain.size_y
            
            self.softmax_iter += 1
            return x_res, y_res
        else:
            Q = np.array([[self.local_terrain[i, j]['mean'] for j in range(self.terrain.size_y)] for i in range(self.terrain.size_x)])
            exp_Q = np.exp(Q / self.softmax_temp)
            
            probabilities = (exp_Q / np.sum(exp_Q)).flatten()
            selected_index = np.random.choice(len(probabilities), p=probabilities)
            
            x_res = selected_index // self.terrain.size_y
            y_res = selected_index % self.terrain.size_y

            self.softmax_iter += 1
            return x_res, y_res

    def _select_position_ucb(self):
        ucb_index = np.zeros((self.terrain.size_x, self.terrain.size_y))
        for i in range(self.terrain.size_x):
            for j in range(self.terrain.size_y):
                if self.local_terrain[i, j]['trials'] == 0:
                    return i, j
                else:
                    ucb_index[i, j] = self.local_terrain[i, j]['mean'] + np.sqrt(2 * np.log(self.trials) / self.local_terrain[i, j]['trials'])
        return np.unravel_index(np.argmax(ucb_index), ucb_index.shape)
        
    def update_knowledge(self, x, y, reward):
        # Met à jour la connaissance de l’agent sur le terrain local
        self.local_terrain[x, y]["mean"] = (self.local_terrain[x, y]["mean"] * self.local_terrain[x, y]["trials"] + reward) / (self.local_terrain[x, y]["trials"] + 1) # Compute the mean
        self.local_terrain[x, y]["trials"] += 1 # Increment the number of trials

    def forward(self):
        x, y = self.select_position()
        reward = self.terrain.get_reward(x, y)
        self.update_knowledge(x, y, reward)

        return reward