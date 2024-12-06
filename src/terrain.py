import numpy as np
import matplotlib.pyplot as plt

class Terrain:
    def __init__(self, sizex: int, sizey: int, mu_scale: int = 10, sigma_scale: int = 5):
        # Initialise le terrain avec la taille (sizex, sizey)
        # Génère aléatoirement les récompenses pour chaque position
        self.size_x = sizex
        self.size_y = sizey
        self.terrain = None
        self.mu_scale = mu_scale
        self.sigma_scale = sigma_scale
        self.generate_terrain()
        self.max_reward = np.max([pos['mu'] for pos in self.terrain.flatten()])

    def generate_terrain(self):
        terrain = np.empty((self.size_x, self.size_y), dtype=object)
        for i in range(self.size_x):
            for j in range(self.size_y):
                mu = np.random.rand() * self.mu_scale
                sigma = np.random.rand() * self.sigma_scale
                terrain[i, j] = {'mu': mu, 'sigma': sigma}
        self.terrain = terrain

    def get_reward(self, x, y):
        # Renvoie la récompense à la position (x, y)
        pos = self.terrain[x,y]
        return max(0, np.random.normal(loc=pos['mu'], scale=pos['sigma']))

    def visualize(self):
        # Visualise le terrain et les récompenses associées
        fig, ax = plt.subplots(figsize=(20, 12))
        for i in range(self.size_x):
            for j in range(self.size_y):
                pos = self.terrain[i, j]
                mu = pos['mu']
                sigma = pos['sigma']
                ax.text(j, i, f'μ={mu:.2f}\nσ={sigma:.2f}', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        ax.set_title('Terrain Rewards')
        ax.set_xticks(np.arange(self.size_y))
        ax.set_yticks(np.arange(self.size_x))
        ax.set_xticklabels(np.arange(self.size_y))
        ax.set_yticklabels(np.arange(self.size_x))
        ax.set_xlim(-0.5, self.size_y - 0.5)
        ax.set_ylim(-0.5, self.size_x - 0.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')

        plt.show()