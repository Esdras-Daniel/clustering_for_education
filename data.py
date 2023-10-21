from sklearn.datasets import make_blobs
from Kmeans import *
from MeanShift import *

np.random.seed(42)

# Geração de dados
def generate_data(n_samples, n_features, centers, cluster_std):
  data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=0)
  return data

NUM_SAMPLES = 300
N_FEATURES = 2
N_CENTERS = 4

# Gera dados
data = generate_data(n_samples=NUM_SAMPLES, n_features=N_FEATURES, centers=N_CENTERS, cluster_std=0.5)

#obj = KMEANS(data_points = data, n_clusters = N_CENTERS, max_iteration = 10, eps = 1e-4)
#obj.show_clustering()

obj = MEANSHIFT(data_points = data, bandwidth = 1.5, max_iteration = 10, eps = 1e-4)
obj.show_clustering()