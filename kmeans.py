import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol 
    
    def fit(self, X):
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        for i in range(self.max_iters):

            self.labels = np.array([self._closest_centroid(x) for x in X])
            

            new_centroids = np.array([X[self.labels == j].mean(axis=0) if len(X[self.labels == j]) > 0 else self.centroids[j]
                                      for j in range(self.k)])
            
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break
            
            self.centroids = new_centroids
        
    def predict(self, X):
        return np.array([self._closest_centroid(x) for x in X])
    
    def _closest_centroid(self, x):
        return np.argmin(np.linalg.norm(self.centroids - x, axis=1))
