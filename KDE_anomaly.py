#! /usr/bin/python

import datetime
import math
import pickle
import os
import sys

import numpy as np
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt

import IFGT

# the file extension for the training and testing data
DATA_EXT = ".npy"

TRAINING_DIR = "data/training/data"
TESTING_DIR = "data/out/data"
OUT_DIR = "results/"

class KDEClassifier:
    """A class to classify points using a kernel density estimate.

    Given a set of training points, the class can classify any test point
    as either probably in the same set as the training points or not.
    """

    def __init__(self, training_data, num_clusters=100, max_multi_index_size=2):
        """Initialization."""

        # general variables

        self.training_data = training_data
        self.covariance = np.cov(training_data.T)
        self.N = len(training_data)
        self.d = len(training_data[0])
        self.h = KDEClassifier.optimal_fixed_bandwidth(self.N, self.d)

        # IFGT variables

        # see the IFGT initialization for more explanation
        # coefficients are for a gaussian kernel
        ifgt_args = dict(
            norm_matrix = np.linalg.inv(self.covariance),
            gaussian_coefficients = np.repeat(1.0/(self.N*(self.h**self.d)*(2*math.pi)**(self.d/2.0)*math.sqrt(np.linalg.det(self.covariance))), self.N),
            exponential_coefficient = 1.0/(2*self.h**2),
            num_clusters = num_clusters,
            max_multi_index_size = max_multi_index_size
        )
        self.ifgt = IFGT.IFGT(self.training_data, **ifgt_args)
        self.ifgt.calculate_coefficients()

        # a cache to store expensive calculations
        self.cache = {}

    @staticmethod
    def optimal_fixed_bandwidth(N, d):
        """The optimal bandwidth for a guassian kernel of N points with dimension d."""

        A = (4.0/(d+2))**(1.0/(d+4))
        return A * N**(-1.0/(d+4))

    def KDE_single(self, x, fixed=True, approx=True):
        """Calculates the fixed-bandwidth KDE for the given point x.

        Just a call to the internal IFGT. If approx is True it
        uses the approximation, otherwise it's exact.
        """

        return self.KDE(x[np.newaxis,:], fixed=fixed, approx=approx)[0]

    def KDE(self, xs, fixed=True, approx=True):
        """Calculates the fixed-bandwidth KDE for the given points xs.

        Just a call to the internal IFGT. If approx is True it
        uses the approximation, otherwise it's exact.
        """

        if fixed:
            if approx:
                return self.ifgt.sum_approx(xs)
            else:
                return self.ifgt.sum_exact(xs)
        else:
            hs = self.adaptive_bandwidths()
            diffs = xs[:,np.newaxis,:] - self.training_data[np.newaxis,:,:]
            exponents = np.exp(-1.0/(2*hs[np.newaxis,:]**2) * np.einsum("lki,ij,lkj->lk", diffs, np.linalg.inv(self.covariance), diffs))
            coeffs = 1.0/(self.N*(hs**self.d)*(2*math.pi)**(self.d/2.0)*math.sqrt(np.linalg.det(self.covariance)))
            return np.einsum("j,ij->i", coeffs, exponents)

    def adaptive_bandwidths(self):
        """Computes the bandwidths for the adaptive KDE."""

        key = "adaptive_bandwidths"
        if key not in self.cache:
            KDE_list = self.KDE_of_training_list(fixed=True, approx=False)
            geom_mean = gmean(KDE_list)
            lambdas = np.power(KDE_list/geom_mean, -0.5)
            self.cache[key] = lambdas * self.h
        return self.cache[key]

    def KDE_of_training_list(self, fixed=True, approx=True):
        """Computes the fixed-bandwidth KDE of every point in the training data.

        The results are cached since it's an expensive operation.
        If approx is True then the approximate sum is used, otherwise
        it's exact.
        """

        if fixed:
            if approx:
                key = "exact_fixed_training_KDEs"
            else:
                key = "approx_fixed_training_KDEs"
        else:
            key = "adaptive_training_KDEs"

        if key not in self.cache:
            self.cache[key] = self.KDE(self.training_data, fixed=fixed, approx=approx)
        return self.cache[key]

    def classify_single(self, x, alpha=0.8, fixed=True, approx=True):
        """Classifies a single point x.

        alpha determines how strict to be. Lower alpha means less strict.
        alpha = 0 means everything is classified as True. Use alpha <= 1 to
        guarantee that all training points are classified as True.

        approx is as in the previous functions.
        """

        return self.classify(x[np.newaxis,:], alpha, approx)[0]

    def classify(self, xs, alpha=0.8, fixed=True, approx=True):
        """Classifies a list of points xs.

        See classify_single(). Returns a list of booleans.
        """

        return self.KDE(xs, fixed=fixed, approx=approx) < alpha * min(self.KDE_of_training_list(fixed=fixed, approx=approx))


def load_npy_file(directory, filename):
    temp_data = None
    try:
        temp_data = np.load(os.path.join(directory, filename))
    except IOError:
        print "failed to read file " + f
    return temp_data

"""
loads the trajectory data from the given directory
the data is grouped by origin (the initial data point for a given ship)
returns a list of [origin, trajectory_data] for each origin
"""
def load_and_group_data(data_dir, maxships=None):
    # a list of [origin, data]
    grouped_data = []

    count = 0
    for f in os.listdir(data_dir):
        if f[-len(DATA_EXT):] == DATA_EXT and "shipdata" in f:
            temp_data = load_npy_file(data_dir, f)
            if temp_data is None:
                continue

            # the origin is the first data point
            temp_origin = temp_data[0,:]

            # add temp_origin to origins if it hasn't been encountered yet
            if not any([np.equal(gr[0], temp_origin).all(0) for gr in grouped_data]):
                grouped_data.append([temp_origin, np.empty((0,4))])

            # add temp_data to the proper data group
            for group in grouped_data:
                if np.equal(group[0], temp_origin).all(0):
                    group[1] = np.vstack((group[1], temp_data))
                    break
            count += 1
        if maxships is not None and count >= maxships:
            break

    return grouped_data

if __name__ == "__main__":
    grouped_KDEs = []
    grouped_training_data = load_and_group_data(TRAINING_DIR)

    for group in grouped_training_data:
        grouped_KDEs.append([group[0], KDEClassifier(group[1])])
        grouped_KDEs[-1][1].ifgt.clusters.plot_clusters()

    for group in grouped_KDEs:
        origin = group[0]
        print "origin: " + str(origin)
        group[1].fixed_KDE_approx_list()

    data_to_test = []
    for f in os.listdir(TESTING_DIR):
        if f[-len(DATA_EXT):] == DATA_EXT and "shipdata" in f:
            temp_data = load_npy_file(TESTING_DIR, f)
            if temp_data is None:
                continue
            data_to_test.append(temp_data)

    ax = plt.figure().add_subplot(111)
    plt.cla()
    if not os.path.exists(os.path.join(OUT_DIR, "times")):
        os.makedirs(os.path.join(OUT_DIR, "times"))
    if not os.path.exists(os.path.join(OUT_DIR, "anomtimes")):
        os.makedirs(os.path.join(OUT_DIR, "anomtimes"))
    shipcount = 0
    makeplot = False
    for ship in data_to_test:
        for group in grouped_KDEs:
            if np.equal(group[0], ship[0]).all(0):
                fixed_anom_bools = np.apply_along_axis(group[1].test_anomaly_fixed_approx, 1, ship)

                if makeplot:
                    normal_colors = ["#00FF00"] * len(ship)
                    anom_colors = ["#FF0000"] * len(ship)

                    fixed_colors = np.where(fixed_anom_bools, anom_colors, normal_colors)

                    ax.scatter(np.transpose(ports)[0], np.transpose(ports)[1], c='g', marker='o')
                    ax.scatter(np.transpose(anomPorts)[0], np.transpose(anomPorts)[1], c='r', marker='o')
                    ax.scatter(ship.T[0], ship.T[1], c=fixed_colors, marker='.')
                    plt.savefig(os.path.join(OUT_DIR, "shipimgfixed" + str(shipcount).zfill(3) + ".png"))
                    plt.cla()

                np.save(os.path.join(OUT_DIR, "anomtimes", "shipanomtimes" + str(shipcount).zfill(3) + ".npy"), np.where(fixed_anom_bools))
                np.save(os.path.join(OUT_DIR, "times", "shiptimes" + str(shipcount).zfill(3) + ".npy"), np.array(range(len(ship))))

                f = open(os.path.join(OUT_DIR, "info.txt"), "a")
                f.write("ship: " + str(shipcount) + "\n")
                f.write("fixed anom times: " + str(np.where(fixed_anom_bools)) + "\n")
                f.close()
        shipcount += 1
