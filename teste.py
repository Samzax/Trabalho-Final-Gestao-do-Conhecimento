import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from kmeans import KMeans
from dbscan import DBSCAN
from som import SOM

X = load_iris().data

k_values = [2, 3, 4, 5]
eps_values = [0.3, 0.5, 0.7]
min_pts_values = [3, 5, 7] 
som_grids = [(2,2), (3,3), (4,4)]
som_sigmas = [0.5, 1.0]
som_learning_rates = [0.3, 0.5]

#K-means
print("Executando K-means")
fig_kmeans, axs_kmeans = plt.subplots(2, 2, figsize=(10, 10)) 
axs_kmeans = axs_kmeans.flatten() 

for i, k in enumerate(k_values):
    kmeans = KMeans(k=k)
    kmeans.fit(X)
    axs_kmeans[i].scatter(X[:,0], X[:,1], c=kmeans.labels, cmap='viridis')
    axs_kmeans[i].set_title(f"K-means: k={k}")
    axs_kmeans[i].set_xlabel("Sepal Length")
    axs_kmeans[i].set_ylabel("Sepal Width")

plt.suptitle("Resultados do K-means com Diferentes Valores de k", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("kmeans_clusters.png", dpi=300)
plt.show()

#DBSCAN 
print("Executando DBSCAN")
fig_dbscan, axs_dbscan = plt.subplots(2, 3, figsize=(15, 10))
axs_dbscan = axs_dbscan.flatten()
plot_idx = 0

for eps in eps_values:
    for min_pts in min_pts_values:
        if plot_idx >= 6:
            break
        dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        dbscan.fit(X)
        
        unique_labels = set(dbscan.labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        axs_dbscan[plot_idx].scatter(X[:,0], X[:,1], c=dbscan.labels, cmap='viridis')
        axs_dbscan[plot_idx].set_title(f"DBSCAN: eps={eps}, minPts={min_pts}")
        axs_dbscan[plot_idx].set_xlabel("Sepal Length")
        axs_dbscan[plot_idx].set_ylabel("Sepal Width")
        plot_idx += 1

plt.suptitle("Resultados do DBSCAN com Diferentes Configurações", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("dbscan_clusters.png", dpi=300)
plt.show()

#SOM
print("Executando SOM")

fig_som, axs_som = plt.subplots(3, 4, figsize=(20, 15))
axs_som = axs_som.flatten()
plot_idx = 0

for grid in som_grids:
    for sigma in som_sigmas:
        for lr in som_learning_rates:
            som = SOM(grid_shape=grid, input_dim=4, sigma=sigma, learning_rate=lr, epochs=100)
            som.fit(X)
            bmus = som.predict(X)

            labels = [i * grid[1] + j for i,j in bmus]
            
            axs_som[plot_idx].scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
            axs_som[plot_idx].set_title(f"SOM: Grid={grid}, σ={sigma}, LR={lr}", fontsize=10)
            axs_som[plot_idx].set_xlabel("Sepal Length")
            axs_som[plot_idx].set_ylabel("Sepal Width")
            plot_idx += 1

plt.suptitle("Resultados do SOM com Diferentes Configurações", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("som_clusters.png", dpi=300)
plt.show()