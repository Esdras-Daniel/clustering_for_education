import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class MEANSHIFT():
    def __init__(self, data_points, bandwidth, max_iteration, eps) -> None:
        self.data_points = data_points
        self.bandwidth = bandwidth
        self.max_iteration = max_iteration
        self.eps = eps
        self.labels = np.zeros(len(data_points))

    # Private function
    def __points_in_bandwidth(self, center_point):
        points_in_bandwidth = []

        for point in self.data_points:
            if (point == center_point).all():
                continue
            else:
                # Compute the distance between the current point and the center one
                distance = np.linalg.norm(point - center_point)

                # Check if the distance is within the bandwidth
                if distance <= self.bandwidth:
                    points_in_bandwidth.append(point)
            
        # Secures that if there is no neighbor point the points_in_bandwidth inserts the point into the list
        if points_in_bandwidth == []:
            points_in_bandwidth = center_point.copy()

        return points_in_bandwidth

    def show_clustering(self):
        # Break flag
        flag = False
        epoch = 1

        # Creating an array that will store the moving data points
        moving_data = self.data_points.copy()

        # Figure
        fig, ax = plt.subplots()

        while not flag:

            # Clean canvas
            ax.clear()
            ax.set_title(f'Bandwidth = {self.bandwidth} - Epoch {epoch}')

            # Original data scatterplot
            scatter = ax.scatter(self.data_points[:, 0], self.data_points[:, 1], c='grey', alpha=0.3)

            # Moving data scatterplot
            scatter = ax.scatter(moving_data[:, 0], moving_data[:, 1], c='red')
            plt.draw()

            # Saving moving_data to verify the stop condition
            prev_moving_data = moving_data.copy()

            # Cycling through each point in moving_data
            for i, point in enumerate(moving_data):

                # Calculating which points in the original data are in the vicinity of the observed point
                neighbor_points = self.__points_in_bandwidth(center_point = point)

                # Calculating the mean
                mean = np.mean(neighbor_points, axis=0)

                # Changing the value of moving points
                moving_data[i] = mean
            
            # Waits for the user to click the image to continue
            plt.waitforbuttonpress()

            # Break conditions
            if np.linalg.norm(prev_moving_data - moving_data) < self.eps:
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
        
        # Drawing the clusters result
        for c, unique_centers in enumerate(np.unique(moving_data, axis=0)):
            for i, point in enumerate(moving_data):
                if (point == unique_centers).all():
                    self.labels[i] = c
 
        # Scatterplot dos dados originais
        ax.clear()
        ax.set_title(f'Final clusters')
        
        scatter = ax.scatter(self.data_points[:, 0], self.data_points[:, 1], c=self.labels)
        
        # Produce a legend with the unique colors from the scatter
        legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
        ax.add_artist(legend)
        plt.draw()
        plt.waitforbuttonpress()
