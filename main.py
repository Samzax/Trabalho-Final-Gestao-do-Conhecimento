
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from kmeans import KMeans
from dbscan import DBSCAN
from som import SOM

data = load_iris()
X = data.data

# K-means
kmeans = KMeans(k=3)
kmeans.fit(X)
plt.scatter(X[:,0], X[:,1], c=kmeans.labels)
plt.title("K-means Clustering")
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.6, min_pts=5)
dbscan.fit(X)
plt.scatter(X[:,0], X[:,1], c=dbscan.labels)
plt.title("DBSCAN Clustering")
plt.show()

# SOM
som = SOM(grid_shape=(3,3), input_dim=4, sigma=1.0, learning_rate=0.5, epochs=100)
som.fit(X)
bmus = som.predict(X)

plt.scatter(X[:,0], X[:,1], c=[i*som.grid_shape[1]+j for i,j in bmus])
plt.title("SOM Clustering")
plt.show()
