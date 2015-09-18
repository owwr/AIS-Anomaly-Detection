#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

import os, sys


"""
Used to make fake ship paths to test the anomaly detection.
Not really needed since we have actual AIS data now.
"""

if len(sys.argv) != 2:
    print """
        mktraj: Simulates normal and anomalous trajectories in 2-dimensional space

        Usage:
            python  mkpath.py   path/to/output/

            - Output will be NumPy arrays (one per vessel)
    """
    print len(sys.argv)
    sys.exit(0)


outdir = sys.argv[1]

## ------------ PARAMETERS ------------ ##
numTraj   = 200                 # Number of trajectories
speedMean = 0.25                 # Mean distance units moved per iteration
speedVar  = 0.03                # Speed variance

probAnomalous = 0.5          # Percentage of ships which are anomalous
probNormalAnom = 0.5
probStopAnom = 0.3
probTurnAroundAnom = 0.2

directionVariance = 0.1         # Variance in ships navigation

meanDeviationTime = 25          # Parameters control when the vessels deviate from course
stdDeviationTime = 8

DISTANCE_THRESH = 0.2

## ------------ LOCATIONS ------------ ##
# - Test locations for now, read in later

# --- Ports --- #
ports = np.array([[6, 4.3], [3.7, -7.4], [-5.5,  0.6], [-7, -3]])

# Set up port adjacency graph
portGraph = np.ones((len(ports), len(ports)))
portGraph[2,3] = 0
portGraph[3,2] = 0
for i in range(len(ports)):
    portGraph[i,i] = 0

# --- Anomalous Ports --- #
anomPorts = np.array([[0, 6], [8, -2]])

## ------------ SIMULATION ------------ ##

if not os.path.exists(os.path.join(outdir, "times")):
    os.makedirs(os.path.join(outdir, "times"))
if not os.path.exists(os.path.join(outdir, "anomtimes")):
    os.makedirs(os.path.join(outdir, "anomtimes"))
if not os.path.exists(os.path.join(outdir, "data")):
    os.makedirs(os.path.join(outdir, "data"))

np.save(os.path.join(outdir, "ports.npy"), ports)
np.save(os.path.join(outdir, "anomports.npy"), anomPorts)

ax = plt.figure().add_subplot(111)
plt.cla()

for shipnum in range(numTraj):
    # Anomalous Ship Check
    is_anomalous = np.random.binomial(1, probAnomalous)
    anom_type = None
    if is_anomalous:
        anom_type = np.digitize([np.random.rand()], np.add.accumulate([probNormalAnom, probStopAnom, probTurnAroundAnom]))[0]

    # Set Origin
    oID = 1 #np.random.choice( range(numPorts), 1 )
    origin = ports[oID]

    target_set = False
    tID = None
    target = None
    while not target_set:
        tID = np.random.randint(len(ports))
        if portGraph[oID, tID] == 1:
            target = ports[tID]
            target_set = True

    # Modify Target for anomalous vessel
    if is_anomalous:
        if anom_type == 0:
            tID = np.random.randint(len(anomPorts))
            anom_target = anomPorts[tID]
        elif anom_type == 1:
            anom_target = target
            stop_length = max(int(np.random.normal(10, 4)), 1)
        else:
            anom_target = origin

    # Begin Journey
    path = [[origin[0], origin[1], 0, 0]]
    pos = np.copy(origin)
    spd = np.random.normal(speedMean, speedVar)

    devTime = int(np.random.normal(meanDeviationTime, stdDeviationTime))
    devTime = max(devTime, 1)

    times = []
    anom_times = []

    atLocation = False
    t = 0
    times.append(t)
    tar = target
    while not atLocation:
        delta = tar - pos                       # Find Direction to Go

        d = np.linalg.norm(delta)

        if d > DISTANCE_THRESH:
            # - Add noise to the motion
            if( directionVariance > 0 ):
                delta[0] += np.random.normal(0, directionVariance) * d
                delta[1] += np.random.normal(0, directionVariance) * d

            delta /= np.linalg.norm( delta )    # Normalize Vector

            # - Apply Movement
            pos += spd * delta
            path.append([pos[0], pos[1], delta[0], delta[1]])
        else:
            break

        t += 1
        times.append(t)
        if is_anomalous and anom_type == 1 and devTime < t < devTime + stop_length:
            anom_times.append(t)
            path.append([path[-1][0], path[-1][1], 0, 0])
            continue
        elif is_anomalous and t > devTime:
            anom_times.append(t)
            tar = anom_target
        else:
            tar = target

    f = open(os.path.join(outdir, "info.txt"), "a")
    f.write("ship: " + str(shipnum).zfill(3) + "\n")
    f.write("anomalous: " + str(is_anomalous) + "\n")
    f.write("path length: " + str(len(path)) + "\n")
    f.write("normal route: " + str(origin) + " -> " + str(target) + "\n")
    if is_anomalous:
        f.write("anomalous type: " + str(anom_type) + "\n")
        f.write("anomalous target: " + str(anom_target) + "\n")
        f.write("deviation time: " + str(devTime) + "\n")
        if anom_type == 1:
            f.write("stop length: " + str(stop_length) + "\n")
    f.write("\n")
    f.close()

    np.save(os.path.join(outdir, "times", "shiptimes" + str(shipnum).zfill(3) + ".npy"), np.array(times))
    np.save(os.path.join(outdir, "anomtimes", "shipanomtimes" + str(shipnum).zfill(3) + ".npy"), np.array(anom_times))

    """
    ax.scatter(np.transpose(ports)[0], np.transpose(ports)[1], c='g', marker='o')
    ax.scatter(np.transpose(anomPorts)[0], np.transpose(anomPorts)[1], c='r', marker='o')
    ax.scatter(np.transpose(path)[0], np.transpose(path)[1], c='b', marker='.')
    plt.savefig(os.path.join(outdir, "shipimg" + str(shipnum).zfill(3) + ".png"))
    plt.cla()
    """

    np.save(os.path.join(outdir, "data", "shipdata" + str(shipnum).zfill(3) + ".npy"), np.array(path))
