import numpy as np


class KMeans():
    def __init__(self, max_iter=50):
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
    X = np.array([
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
    ])

    kmeans = KMeans()
    kmeans.fit(X, centroids=[0, 5])
    y = kmeans.predict(X)
    print(y)


if __name__ == '__main__':
    main()
