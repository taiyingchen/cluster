from sys import stdin
from math import pi, sqrt, exp
from .utils import *


class GMM():
    def __init__(self, max_iter=100):
        self.max_iter = max_iter

    def fit(self, X, clusters):
        self.num_clusters = len(clusters)
        self.num_dimensions = len(clusters[0])
        self.num_data = len(X)

        if len(X) == 0 or len(clusters) == 0:
            raise ValueError('Found empty data in X or clusters')

        # Initialize cluster prior probability (w) to 1 / n
        # Where n is the total number of data points
        self.w = [1 / self.num_data for i in range(self.num_clusters)]
        # Initialize mean (mu) to given cluster centroid
        self.mu = clusters.copy()
        # Initialize covariance matrix (sigma) to identity matrix
        covar_matrix = [[1. if i == j else 0. for j in range(
            self.num_dimensions)] for i in range(self.num_dimensions)]
        self.sigma = [covar_matrix for i in range(self.num_clusters)]
        # Joint probability (wij) of xi and its cluster cj
        self.joint_prob = [
            [0. for j in range(self.num_clusters)] for i in range(self.num_data)]

        self.em_algo(X)

    def gaussian_prob(self, xi, j):
        """Return Gaussian probability from cluster j
        """
        invert_sigma = invert_2d_matrix(self.sigma[j])
        x_minus_mu = sub_vector(xi, self.mu[j])
        numerator = mul_vector_2d_matrix(x_minus_mu, invert_sigma)
        numerator = dot_vector(numerator, x_minus_mu)
        numerator = exp(-1/2 * numerator)
        denominator = sqrt((2*pi)**self.num_dimensions *
                           det_2d_matrix(self.sigma[j]))
        return numerator / denominator

    def em_algo(self, X):
        """Expectation Maximization Algorithm
        """
        for _ in range(self.max_iter):
            self.expectation(X)
            self.maximization(X)

    def expectation(self, X):
        """Expectation step of EM Algorithm
        """
        for i in range(self.num_data):
            denominator = 0.
            # Calcuate numerator and accumerate denominator
            for j in range(self.num_clusters):
                numerator = self.w[j] * self.gaussian_prob(X[i], j)
                self.joint_prob[i][j] = numerator
                denominator += numerator
            # Divide every element by denominator
            for j in range(self.num_clusters):
                self.joint_prob[i][j] /= denominator

    def maximization(self, X):
        """Maximization step of EM Algorithm
        """
        for j in range(self.num_clusters):
            sum_joint_prob = 0.
            sum_mu = [0. for i in range(self.num_dimensions)]
            sum_sigma = [[0. for j in range(self.num_dimensions)]
                         for i in range(self.num_dimensions)]

            for i in range(self.num_data):
                sum_mu = add_vector(
                    sum_mu, [self.joint_prob[i][j]*xij for xij in X[i]])

                x_minus_mu = sub_vector(X[i], self.mu[j])
                sigma = mul_vector(x_minus_mu, x_minus_mu)
                sigma = mul_scalar_2d_matrix(self.joint_prob[i][j], sigma)
                sum_sigma = add_2d_matrix(sum_sigma, sigma)

                sum_joint_prob += self.joint_prob[i][j]

            self.w[j] = sum_joint_prob / self.num_data
            self.mu[j] = [v/sum_joint_prob for v in sum_mu]
            self.sigma[j] = div_scalar_2d_matrix(sum_joint_prob, sum_sigma)

    def predict(self, X):
        y = []
        for xi in X:
            max_prob = -float('inf')
            yi = -1
            for j in range(self.num_clusters):
                prob = self.gaussian_prob(xi, j)
                if prob > max_prob:
                    max_prob = prob
                    yi = j
            y.append(yi)
        return y


def main():
    X = [[8.98320053625, -2.08946304844],
         [2.61615632899, 9.46426282022],
         [1.60822068547, 8.29785986996],
         [8.64957587261, -0.882595891607],
         [1.01364234605, 10.0300852081],
         [1.49172651098, 8.68816850944],
         [7.95531802235, -1.96381815529],
         [0.527763520075, 9.22731148332],
         [6.91660822453, -3.2344537134],
         [6.48286208351, -0.605353440895]]
    clusters = [[3.35228193353, 6.27493570626], [6.76656276363, 6.54028732984]]

    gmm = GMM(max_iter=100)
    gmm.fit(X, clusters)
    y = gmm.predict(X)
    print(y)


if __name__ == '__main__':
    main()
