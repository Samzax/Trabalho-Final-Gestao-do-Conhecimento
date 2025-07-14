import numpy as np

class SOM:
    def __init__(self, grid_shape=(3, 3), input_dim=4, sigma=1.0, learning_rate=0.5, epochs=100):
        self.grid_shape = grid_shape
        self.input_dim = input_dim
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(grid_shape[0], grid_shape[1], input_dim)
    
    def fit(self, X):
        for epoch in range(self.epochs):
            for x in X:
                bmu_idx = self._find_bmu(x)
                self._update_weights(x, bmu_idx, epoch)
    
    def _find_bmu(self, x):
        dists = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(dists), self.grid_shape)
    
    def _update_weights(self, x, bmu_idx, epoch):
        lr = self.learning_rate * np.exp(-epoch / self.epochs)
        sigma = self.sigma * np.exp(-epoch / self.epochs)
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                dist_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                influence = np.exp(-dist_to_bmu**2 / (2 * sigma**2))
                self.weights[i, j] += influence * lr * (x - self.weights[i, j])
    
    def predict(self, X):
        return np.array([self._find_bmu(x) for x in X])
