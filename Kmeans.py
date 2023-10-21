import numpy as np
import matplotlib.pyplot as plt

class KMEANS():
    def __init__(self, data_points, n_clusters, max_iteration, eps) -> None:
        self.data_points = data_points
        self.n_clusters = n_clusters
        self.max_iteration = max_iteration
        self.eps = eps
    
    def show_clustering(self):
        # Break flag
        flag = False
        epoch = 1

        # Figure
        fig, ax = plt.subplots()

        # Initialize centroids
        centroids = self.data_points[np.random.choice(self.data_points.shape[0], size=self.n_clusters, replace=False)]

        while not flag:
            
            # Clean canvas
            ax.clear()
            ax.set_title(f'Epoch {epoch}')

            # Classify each data point to its cluster
            clusters = [[] for _ in range(self.n_clusters)]
            for point in self.data_points:
                closest = min(range(self.n_clusters), key=lambda i: np.linalg.norm(point - centroids[i]))
                clusters[closest].append(point)

             # Create the label list for the data points
            labels = np.zeros(len(self.data_points))
            for i in range(self.n_clusters):
                for j in range(len(clusters[i])):
                    point = clusters[i][j]
                    point_idx = np.where(self.data_points == point)[0][0]
                    labels[point_idx] = i
            
            # First Plot (plot all data points related to its cluster)  
            ax.scatter(self.data_points[:, 0], self.data_points[:, 1], c=labels)
            ax.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='*')
            
            # Wait for the button press to go to next fase of the Kmeans clustering method
            plt.draw()
            plt.waitforbuttonpress()

            # Copy of previous centroids
            prev_centroids = [center.copy() for center in centroids]

            # Update centroids
            for i in range(self.n_clusters):
                centroids[i] = np.mean(clusters[i], axis=0)

            # Plot new centroids
            ax.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x')

            # Plot line between new centroids and the previous centroids to show the deslocation
            for i in range(self.n_clusters):
                ax.plot([prev_centroids[i][0], centroids[i][0]],
                        [prev_centroids[i][1], centroids[i][1]],
                        color='black')

            plt.draw()
            plt.waitforbuttonpress()

            # Break conditions
            variance = sum(np.linalg.norm(centroids - prev_centroids, axis = 0))

            if variance < self.eps:
                ax.set_title(f'Convergence reached after {epoch} epochs!')
                plt.draw()
                flag = True
                plt.waitforbuttonpress()

            epoch += 1

            if epoch > self.max_iteration:
                ax.set_title(f'Maximum number of iterations reached!')
                plt.draw()
                flag = True
                plt.waitforbuttonpress()