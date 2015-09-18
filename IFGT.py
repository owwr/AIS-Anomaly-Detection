import random
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial


def graded_lexicographic(n, deg):
    """The exponents of a n-element monomial of degree deg in graded lexicographic order.

    Calculated recursively.
    """

    if deg == 0:
        return [[0]*n]
    if n == 1:
        return [[deg]]

    totlst = []
    for i in range(deg, -1, -1):
        lst = graded_lexicographic(n-1, deg-i)
        for item in lst:
            totlst.append([i] + item)

    return totlst


def graded_lexicographic_up_to(n, deg):
    """The exponents of a n-element monomial of degree up to deg (inclusive) in graded lexicographic order."""

    totlst = []
    for i in range(deg+1):
        totlst.extend(graded_lexicographic(n, i))

    return totlst


def multinomial_coefficient(n, alpha):
    """The multinomial coefficient of the multi-index alpha."""

    return factorial(n, exact=True)/multi_index_factorial(alpha)


def multi_index_factorial(alpha):
    """The factorial of the multi-index alpha."""

    ans = 1
    for a in alpha:
        ans *= factorial(a, exact=True)
    return ans


class FarthestPointClustering:
    """Data structure to form a given set of points into clusters."""

    def __init__(self, data, num_clusters=50, norm_matrix=None):
        """Initialization."""

        self.data = data
        self.N = len(data)
        self.d = len(data[0])
        self.num_clusters = num_clusters
        self.norm_matrix = np.eye(self.d) if norm_matrix is None else norm_matrix

        # the centers of the clusters
        self.centers = None
        # a list of which cluster the points belong to
        # for example, nrst_cntrs_indxs[j] = k means data[j] is closest to centers[k]
        self.nearest_centers_indexes = None
        # the maximum distance from a center to any point in its cluster
        self.cluster_radii = None

        self.form_clusters()

    def _norm(self, point):
        """Calculate the norm of a point using the norm matrix."""

        return np.sqrt(point.dot(self.norm_matrix).dot(point))

    def _norm_of_list(self, points):
        """Calculate the norm of a list of points using the norm matrix."""

        return np.sqrt(np.einsum("ki,ij,kj->k", points, self.norm_matrix, points))

    def add_point(self, point):
        """Add a point to the data structure."""

        self.data = np.append(self.data, point)
        self.N += 1

        self.nearest_centers_indexes = np.append(self.nearest_centers_indexes, np.argmin(self._norm_of_list(point - self.centers)))
        nearest_center = self.centers[self.nearest_centers_indexes[-1]]
        old_radius = self.cluster_radii[self.nearest_centers_indexes[-1]]
        self.cluster_radii[self.nearest_centers_indexes[-1]] = max(old_radius, self._norm(point - nearest_center))

    def form_clusters(self):
        """Group all the points into clusters."""

        # start with one randomly-chosen centers
        self.centers = [random.choice(self.data)]
        dists_to_centers = self._norm_of_list(self.data - self.centers[0])
        self.nearest_centers_indexes = np.zeros(self.N, dtype=int)

        # each new center is the point farthest away from the current centers
        for i in range(1, self.num_clusters):
            max_dist_point = self.data[np.argmax(dists_to_centers)]
            self.centers.append(max_dist_point)

            new_dists = self._norm_of_list(self.data - max_dist_point)
            self.nearest_centers_indexes = np.where(new_dists < dists_to_centers, np.repeat(i, self.N), self.nearest_centers_indexes)
            dists_to_centers = np.minimum(new_dists, dists_to_centers)

        self.centers = np.array(self.centers)
        self.cluster_radii = np.apply_along_axis(lambda c: np.max(self._norm_of_list(c - self.data[self.get_cluster_indexes(c)])), 1, self.centers)

    def get_cluster_indexes(self, center):
        """Returns the indexes of the points belonging to the cluster with the given center."""

        return np.where((self.centers[self.nearest_centers_indexes] == center).all(1))

    def get_centers_near_point(self, point, radius=np.Infinity):
        """Get a list of centers within radius of the given point, sorted by distance (nearest first)."""

        dists_to_centers = self._norm_of_list(point - self.centers)
        sort_indexes = np.argsort(dists_to_centers)
        sorted_centers = self.centers[sort_indexes]
        sorted_dists = dists_to_centers[sort_indexes]
        return sorted_centers[np.where(sorted_dists < radius)]

    def cluster_plot(self):
        """Plot the clusters and cluster centers.

        The diminsionality of the data must be at least 2. If it's greater than 2,
        only the first two dimensions are plotted.
        """

        center_colors = np.random.rand(self.num_clusters, 3)
        point_colors = center_colors[self.nearest_centers_indexes]

        ax = plt.figure().add_subplot(111)
        ax.scatter(np.transpose(self.data)[0], np.transpose(self.data)[1], c=point_colors)
        ax.scatter(np.transpose(self.centers)[0], np.transpose(self.centers)[1], c='k', marker='x')
        plt.show()


class IFGT:
    """Class to evaluate sums of gaussians.

    Calculates sum_{i=1}^N qi exp(-a (yj-xi).M.(yj-xi) )
    for a list of points yj, where
    xi is a given list of points
    M is the norm_matrix (should be symmetric and positive-semidefinite, e.g. the covariance of the xs)
    qi are the gaussian_coefficients
    a is the exponential_coefficient (should be postitive, e.g. 1/(2h^2), so that the exponential decays)

    The class provides methods for either calculating the sum exactly or using the Improved Fast Gauss Transform
    to approximate the sum.
    """

    def __init__(self, xs, norm_matrix=None, gaussian_coefficients=None, exponential_coefficient=None, num_clusters=50, max_multi_index_size=2):
        """Initialization."""

        # general variables
        # see the class docstring for more info

        self.xs = xs
        self.N = len(xs)
        self.d = len(xs[0])
        self.norm_matrix = np.eye(d) if norm_matrix is None else norm_matrix
        self.gaussian_coefficients = np.repeat(1.0, self.N) if gaussian_coefficients is None else gaussian_coefficients
        self.exponential_coefficient = 1.0 if exponential_coefficient is None else exponential_coefficient

        # cluster variables

        self.num_clusters = num_clusters
        self.clusters = FarthestPointClustering(xs, num_clusters, self.norm_matrix)

        # expansion coefficient variables

        self.max_multi_index_size = max_multi_index_size
        self.expansion_coefficients = None
        # precalculate the needed multi-indexes
        self.multi_indexes = np.array(graded_lexicographic_up_to(self.d, self.max_multi_index_size), dtype=np.int)

    def calculate_coefficients(self):
        """Calculate the coefficients for the approximate expansion."""

        self.expansion_coefficients = np.empty((self.num_clusters, len(self.multi_indexes)))

        alpha_coeffs = np.apply_along_axis(lambda a: np.power(2*self.exponential_coefficient, np.sum(a))/multi_index_factorial(a), 1, self.multi_indexes)
        for c in range(self.num_clusters):
            cluster_indexes = self.clusters.get_cluster_indexes(self.clusters.centers[c])
            points = self.xs[cluster_indexes]
            coeffs = self.gaussian_coefficients[cluster_indexes]

            # warning: excessive use of np.einsum
            diffs = points - self.clusters.centers[c]
            transformed_diffs = np.einsum("ij,kj->ki", self.norm_matrix, diffs)
            exponents = coeffs * np.exp(-self.exponential_coefficient * np.einsum("ki,ij,kj->k", diffs, self.norm_matrix, diffs))
            prod = np.product(np.power(transformed_diffs[:,np.newaxis,:], self.multi_indexes[np.newaxis,:,:]), axis=2)
            self.expansion_coefficients[c] = np.einsum("i,j,ij->j", exponents, alpha_coeffs, prod)

    def sum_approx_single(self, y):
        """Approximate the sum of gaussians at the single point y."""

        return self.sum_approx(y[np.newaxis,:])[0]

    def sum_approx(self, ys):
        """Approximate the sum of gaussians at points ys.

        Returns a list of values, one for each of the ys.
        """

        # we need the coefficients to do the approximation
        if self.expansion_coefficients is None:
            self.calculate_coefficients()

        diffs = ys[:,np.newaxis,:] - self.clusters.centers[np.newaxis,:,:]
        powers = np.product(np.power(diffs[:,:,np.newaxis,:], self.multi_indexes[np.newaxis,np.newaxis,:,:]), axis=3)
        exponents = np.exp(-self.exponential_coefficient * np.einsum("lki,ij,lkj->lk", diffs, self.norm_matrix, diffs))
        return np.einsum("ijk,ijk,ijk->i", self.expansion_coefficients[np.newaxis,:,:], exponents[:,:,np.newaxis], powers)

    def sum_exact_single(self, y):
        """Exactly evaluate the sum of gaussians at the single point y."""

        return self.sum_exact(y[np.newaxis,:])[0]

    def sum_exact(self, ys):
        """Exactly evaluate the sum of gaussians at points ys.

        Returns a list of values, one for each of the ys.
        """

        diffs = ys[:,np.newaxis,:] - self.xs[np.newaxis,:,:]
        exponents = np.exp(-self.exponential_coefficient * np.einsum("lki,ij,lkj->lk", diffs, self.norm_matrix, diffs))
        return np.einsum("j,ij->i", self.gaussian_coefficients, exponents)


if __name__ == "__main__":
    # testing approximate vs exact sum

    N = int(10**4)
    d = 4
    clusters = 100
    multi_size = 2

    for ypow in range(2, 9):
        points = np.random.rand(N, d)
        ys = np.random.rand(int(10**(ypow/2.0)), d)
        coeffs = np.random.rand(N)

        ifgt = IFGT(points, np.identity(d), coeffs, 1, clusters, multi_size)

        print "N: " + str(N)
        print "clusters: " + str(clusters)
        print "multi_size: " + str(multi_size)
        print "ys: " + str(len(ys))

        start = time.time()
        ifgt.calculate_coefficients()
        print "time to calculate coeffs: " + str(time.time() - start)

        start = time.time()
        s1 = ifgt.sum_exact(ys)
        print "time for exact sum: " + str(time.time() - start)

        start = time.time()
        s2 = ifgt.sum_approx(ys)
        print "time for approx sum: " + str(time.time() - start)

        print "error: " + str(np.mean((s1-s2)/s1)) + "\n"
