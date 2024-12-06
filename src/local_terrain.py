import numpy as np

class LocalTerrain:
    def __init__(self, size_x, size_y):
        self.terrain = self._generate_local_terrain(size_x, size_y)
        
    def _generate_local_terrain(self, size_x, size_y):
        terrain = np.empty((size_x, size_y), dtype=object)
        for i in range(size_x):
            for j in range(size_y):
                terrain[i, j] = {'trials': 0, 'mean': 0}
        return terrain
    
    def __getitem__(self, index):
        return self.terrain[index]