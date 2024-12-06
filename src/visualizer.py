import numpy as np
import matplotlib.pyplot as plt

from .agent import Agent

class Visualizer:
    def __init__(self, agent: Agent):
        # Initialise le visualiseur avec l’agent
        self.agent = agent

    def plot_performance(self, performance_data):
        # Trace les performances de l’agent au fil du temps
        plt.plot(performance_data)
        plt.title('Agent Performance Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.show()

    def draw_terrain_knowledge(self):
        # Dessine la connaissance actuelle du terrain par l’agent
        mean_values = np.array([[self.agent.local_terrain[i, j]["mean"] for j in range(self.agent.terrain.size_y)] 
                      for i in range(self.agent.terrain.size_x)])
        plt.imshow(mean_values, cmap='hot', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label('Average Gain')
        plt.title(f'Terrain Knowledge Heatmap of Agent {self.agent.class_type}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
        
    def draw_diff_agent_knowledge_ground_truth(self):
        # Dessine la différence entre la connaissance de l’agent et la vérité terrain
        mu_values = np.array([[self.agent.terrain.terrain[i, j]["mu"] for j in range(self.agent.terrain.size_y)] 
                      for i in range(self.agent.terrain.size_x)])
        mean_values = np.array([[self.agent.local_terrain[i, j]["mean"] for j in range(self.agent.terrain.size_y)] 
                      for i in range(self.agent.terrain.size_x)])
        diff = abs(mu_values - mean_values)
        plt.imshow(diff, cmap='hot', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label('Difference in Gain')
        plt.title(f'Difference in Agent {self.agent.class_type} Knowledge and Ground Truth')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
        
    def draw_regret(self, performance_data):
        n = len(performance_data)
        regret_data = [self.agent.terrain.max_reward - performance_data[i] for i in range(n)]
        # Trace le regret de l’agent au fil du temps
        plt.plot(np.cumsum(performance_data), label='Cumulative Performance')
        plt.plot(np.cumsum(regret_data), label='Cumulative Regret')
        plt.legend()
        plt.title(f'Agent {self.agent.class_type} Regret Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Regret')
        plt.grid(True)
        plt.show()
    
    def draw_mse(mse_data):
        plt.plot(mse_data[0], label='Uniform MSE')
        plt.plot(mse_data[1], label='Epsilon Greedy MSE')
        plt.plot(mse_data[2], label='Softmax MSE')
        plt.plot(mse_data[3], label='UCB MSE')
        plt.legend()

        plt.title('MSE Comparison for different Agents class_types')
        plt.xlabel('Steps')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()