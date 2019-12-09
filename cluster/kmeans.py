import numpy as np
import matplotlib.pyplot as plt


class KMeans():
    def __init__(self, max_iter=20):
        self.max_iter = max_iter

    def nearest_cluster(self, xi):
        min_dist = float('inf')
        cluster = -1
        for j in range(self.num_clusters):
            dist = np.linalg.norm(xi - self.clusters[j])
            if dist < min_dist:
                min_dist = dist
                cluster = j
        return cluster

    def fit(self, X, centroids):
        self.num_data = len(X)
        self.num_dimensions = len(X[0])
        self.num_clusters = len(centroids)

        self.clusters = [X[i] for i in centroids]
        self.data_clusters = [0 for i in range(self.num_data)]

        for _ in range(self.max_iter):
            print(f'Iteration {_} --------')
            print(self.data_clusters)
            print(self.clusters)

            for i in range(self.num_data):
                self.data_clusters[i] = self.nearest_cluster(X[i])

            for j in range(self.num_clusters):
                self.clusters[j] = np.mean(
                    [X[i] for i, cluster in enumerate(self.data_clusters) if cluster == j], axis=0)

    def predict(self, X):
        y = []
        for xi in X:
            y.append(self.nearest_cluster(xi))
        return y


def main():
    X = [
        [1, 1],
        [1, 3],
        [1, 4],
        [1, 5],
        [0, 4],
        [2, 4],
        [3, 4],
        [4, 4],
        [5, 4],
        [4, 3],
        [4, 5]
    ]
    X = np.array(X)

    kmeans = KMeans()
    kmeans.fit(X, centroids=[0, 5])
    y = kmeans.predict(X)

    X_plot = {}
    for i, cluster in enumerate(y):
        if cluster not in X_plot:
            X_plot[cluster] = []
        X_plot[cluster].append(X[i])

    for cluster in X_plot:
        X_plot[cluster] = np.array(X_plot[cluster])

    plt.scatter(X_plot[0][:, 0], X_plot[0][:, 1], c='b')
    plt.scatter(X_plot[1][:, 0], X_plot[1][:, 1], c='r')
    plt.show()


if __name__ == '__main__':
    main()
