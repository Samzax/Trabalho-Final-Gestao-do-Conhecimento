import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
    
    def fit(self, X):
        n = len(X)
        self.labels = np.full(n, -1)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_pts:
                self.labels[i] = -1 
            else:
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            idx = neighbors[i]
            if not visited[idx]:
                visited[idx] = True
                new_neighbors = self._region_query(X, idx)
                if len(new_neighbors) >= self.min_pts:
                    neighbors += new_neighbors
            if self.labels[idx] == -1:
                self.labels[idx] = cluster_id
            i += 1
    
    def _region_query(self, X, idx):
        dists = np.linalg.norm(X - X[idx], axis=1)
        return list(np.where(dists < self.eps)[0])
