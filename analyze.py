import numpy as np
import os
import sys

DATA_EXT = ".npy"

"""
Tests the accuracy of the anomalous ship detection.
"""

def load_and_group_times(directory):
    times = {}
    anomtimes = {}
    for f in os.listdir(os.path.join(directory, "times")):
        if f[-len(DATA_EXT):] == DATA_EXT:
                shipnum = int(f[-len(DATA_EXT)-3:-len(DATA_EXT)])
                times[shipnum] = np.load(os.path.join(directory, "times", f))
    for f in os.listdir(os.path.join(directory, "anomtimes")):
        if f[-len(DATA_EXT):] == DATA_EXT:
                shipnum = int(f[-len(DATA_EXT)-3:-len(DATA_EXT)])
                anomtimes[shipnum] = np.load(os.path.join(directory, "anomtimes", f))

    return times, anomtimes

if __name__ == "__main__":

    actual_times, actual_anom_times = load_and_group_times(sys.argv[1])
    fast_times, fast_anom_times = load_and_group_times(sys.argv[2])

    num_ships = len(actual_times)

    fast_tot = 0
    fast_true_positive = 0
    fast_true_negative = 0
    fast_false_positive = 0
    fast_false_negative = 0

    for ship in range(num_ships):
        for time in actual_times[ship]:
            fast_tot += 1
            if time in actual_anom_times[ship]:
                if time in fast_anom_times[ship][0]:
                    fast_true_positive += 1
                else:
                    fast_false_negative += 1
            else:
                if time in fast_anom_times[ship][0]:
                    fast_false_positive += 1
                else:
                    fast_true_negative += 1

    print "total times: ".ljust(20) + str(fast_tot)
    print "total correct: ".ljust(20) + str(fast_true_positive + fast_true_negative)
    print "total incorrect: ".ljust(20) + str(fast_false_positive + fast_false_negative)
    print "correct percent: ".ljust(20) + str(100*(fast_true_positive + fast_true_negative)/fast_tot) + "%"
    print "true positive: ".ljust(20) + str(fast_true_positive)
    print "false positive: ".ljust(20) + str(fast_false_positive)
    print "true negative: ".ljust(20) + str(fast_true_negative)
    print "false negative: ".ljust(20) + str(fast_false_negative)
