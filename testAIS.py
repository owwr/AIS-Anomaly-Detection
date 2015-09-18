import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches
import numpy as np
import cPickle as pickle
import requests
import sklearn.cluster
import scipy
import sys
import time
from mpl_toolkits.basemap import Basemap

from KED_anomaly import KDEClassifier
import IFGT

"""
A variety of methods to work with the AIS data. Anomaly detection was successful as long as
you choose a polygon that only contains one path. Tried to get route detection working but
it wasn't successful, the AIS data is not very accurate with regards to velocity.
"""

def store_AIS_points(filename="points.pickle"):
    """Store points from AIS stream."""

    req_url = "http://features.dev.eagle-ow.com:8080/api/search/ais_position"
    total_items = requests.get(req_url + "?limit=0").json()["totalItems"]
    print total_items

    points = []
    parsed = requests.get(req_url + "?limit={}".format(total_items)).json()
    print parsed
    for item in parsed["items"]:
        coords = item["geometry"]["coordinates"]
        cog = item["properties"]["cog"]
        sog = item["properties"]["sog"]
        points.append([coords[0], coords[1], cog, sog])

    with open(filename, "wb") as f:
        pickle.dump(points, f)

def store_all_AIS(filename="all.pickle", split=10000):
    """Store raw data from AIS stream."""

    req_url = "http://features.dev.eagle-ow.com:8080/api/search/ais_position"
    total_items = requests.get(req_url + "?limit=0").json()["totalItems"]

    print total_items

    items = []

    for i in range(0,total_items,split):
        print i
        parsed = requests.get(req_url + "?offset={}&limit={}".format(i, split)).json()
        items += parsed["items"]

    print len(items)

    with open(filename, "w") as f:
        pickle.dump(items, f)

def AIS_to_point(item):
    """Convert an AIS datum to (x, y, vx, vy)."""

    coords = item["geometry"]["coordinates"] # longitude/latitude
    cog = np.pi/180 * item["properties"]["cog"]/10 # radians clockwise from north
    sog = item["properties"]["sog"]/10 # knots
    return np.array([
        coords[0], # x
        coords[1], # y
        sog*np.sin(cog), # vx
        sog*np.cos(cog) # vy
    ])

def create_path(start, end, speedmean, speedvar, thetavar):
    """Simulate a ship moving from start to end."""

    path = []
    p = start[:]
    while np.linalg.norm(p - end) > 0.1:
        direction = end - p
        speed = max(0.001, np.random.normal(speedmean, speedvar))
        theta = np.random.normal(np.arctan2(direction[1], direction[0]), thetavar)
        delta = speed*np.array([np.cos(theta), np.sin(theta)])
        path.append([p[0], p[1], delta[0], delta[1]])
        p += delta
    return path

def create_long_path(points, speedmean, speedvar, thetavar):
    path = []
    for i in range(len(points)-1):
        a = points[i]
        b = points[i+1]
        path.extend(create_path(a, b, speedmean, speedvar, thetavar))
    return path

def get_basemap(minx, maxx, miny, maxy):
    m = Basemap(projection="cyl", llcrnrlon=minx, urcrnrlon=maxx, llcrnrlat=miny, urcrnrlat=maxy, resolution="i")
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='coral',lake_color='white')
    # m.drawparallels(np.arange(miny,maxy,(maxy-miny)/3), labels=[True,True,False,False])
    # m.drawmeridians(np.arange(minx,maxx,(maxx-minx)/3), labels=[False,False,True,True])
    m.drawmapboundary(fill_color='white')
    return m

def test_path():
    """Create a fake path and test it for anomalies using KDEClassifier."""

    with open("points.pickle", "r") as f:
        points = pickle.load(f)

    # all points outside of poly are ignored
    """
    poly = np.array([
            [115.707, 7.9],
            [116.509, 7.727],
            [118.295, 10.3057],
            [118.915, 13.4038],
            [118.21, 13.4038],
            [117.652, 10.667],
            [115.707, 7.9]
            ])
    """
    poly = np.array([
            [106.146, 4.0428],
            [114.046, 12.1145],
            [111.592, 14.1508],
            [105.508, 4.06733],
            [106.146, 4.0428]
            ])
    """
    poly = np.array([
        [113.531, 11.356],
        [113.531, 9.2452],
        [116.328, 9.2452],
        [116.328, 11.356],
        [113.531, 11.356]
    ])
    """
    raw_training_data = np.array([[p[0], p[1], p[3]*np.sin(p[2]/10.0*np.pi/180)/600, p[3]*np.cos(p[2]/10.0*np.pi/180)/600] for p in points])
    path = matplotlib.path.Path(poly[:-1], closed=True)
    training_data = raw_training_data
    # crop using poly
    training_data = training_data[np.apply_along_axis(lambda x: False if path.contains_point([x[0], x[1]])==0 else True, 1, raw_training_data)]
    # select stationary points
    # training_data = training_data[np.where(training_data[:,2]**2+training_data[:,3]**2 < 10**-2)]
    kde = KDEClassifier(training_data)

    minx = min(training_data[:,0])
    miny = min(training_data[:,1])
    maxx = max(training_data[:,0])
    maxy = max(training_data[:,1])

    """
    start = np.array([116.103, 7.87])
    middle = np.array([117.97, 10.47])
    anom1 = np.array([117.67, 11.423])
    anom2 = np.array([119.221, 10.5673])
    anom3 = np.array([116.402, 10.8365])
    normal1 = np.array([117.786, 10.0526])
    normal2 = np.array([118.224, 11.09])
    anom4 = np.array([118.627, 9.99])
    end = np.array([118.604, 13.234])
    testing_data = np.array(create_long_path([start, normal1, anom4, normal2, end], 0.15, 0.03, 0.06))
    """

    # create a fake path
    start = np.array([106.085, 4.458])
    end = np.array([112.875, 12.623])
    normal1 = np.array([109.655, 8.7621])
    normal2 = np.array([111.436, 10.8496])
    normal3 = np.array([108.243, 7.10426])
    anom1 = np.array([108.888, 11.248])
    anom2 = np.array([108.12, 4.0956])
    anom3 = np.array([106.505, 9.48133])
    testing_data = np.array(create_long_path([end, start], 0.15, 0.03, 0.06))

    # classify!
    fixed_anom_bools = kde.classify(testing_data, fixed=True, approx=True)

    # plot results

    m = get_basemap(minx-1, maxx+1, miny-1, maxy+1)
    xpts, ypts = m(training_data[:,0], training_data[:,1])
    us, vs = training_data[:,2], training_data[:,3]
    m.scatter(xpts, ypts, c="k", marker=".", alpha=0.3)
    # m.quiver(xpts, ypts, us, vs, alpha=0.3)

    # m.plot(poly[:,0], poly[:,1])

    xs, ys = m(testing_data[:,0], testing_data[:,1])
    # us, vs = testing_data[:,2], testing_data[:,3]
    colors = np.where(fixed_anom_bools, "r", "g")
    m.scatter(xs, ys, c=colors, marker="8", s=100)
    # m.quiver(xs, ys, us, vs, color=colors, scale=3)

    # m.hexbin(training_data[:,0], training_data[:,1])

    plt.show()

def load_points(filename="all.pickle"):
    with open("all.pickle", "r") as f:
        items = pickle.load(f)

    points = []

    for item in items:
        points.append(AIS_to_point(item))

    return np.array(points)

if __name__ == "__main__":
    #store_all_AIS(split=200000)
    #sys.exit(0)

    # test_path()

    points = load_points()

    print "processed file"

    poly = np.array([
        [91.0695, -6.4],
        [91.0695, -19.542],
        [120.211, -19.542],
        [120.211, -6.4],
        [91.0695, -6.4]
    ])
    path = matplotlib.path.Path(poly, closed=True)

    # points = points[-200000:,:2]
    points = points[np.apply_along_axis(lambda x: False if path.contains_point([x[0], x[1]])==0 else True, 1, points)]
    points = points[np.where(points[:,2]**2+points[:,3]**2 == 0)][:,:2]
    print points.shape

    invcov = np.linalg.inv(np.cov(points.T))
    # metric = sklearn.neighbors.DistanceMetric.get_metric("mahalanobis", VI=invcov)
    def metric(x, y):
        # why is this needed
        if len(x) == 10:
            return 0
        return np.sqrt(np.einsum("i,ij,j", x-y, invcov, x-y))
    # metric = lambda foo, bar: np.sqrt((foo-bar).dot(foo-bar))
    labels = sklearn.cluster.DBSCAN(eps=0.3, metric="euclidean").fit_predict(points[:,:2])

    print labels
    unique_labels = list(set(labels))
    print len(unique_labels)

    cluster_colors = np.random.rand(len(unique_labels), 3)
    cluster_colors_dict = {}
    for i in range(len(unique_labels)):
        if unique_labels[i] == -1:
            cluster_colors_dict[-1] = np.array([0,0,0])
            continue
        cluster_colors_dict[unique_labels[i]] = cluster_colors[i]
    point_colors = np.array([cluster_colors_dict[labels[i]] for i in range(len(labels))])


    minx = min(points[:,0])-1
    miny = min(points[:,1])-1
    maxx = max(points[:,0])+1
    maxy = max(points[:,1])+1
    m = get_basemap(minx, maxx, miny, maxy)
    xs, ys = m(points[:,0], points[:,1])
    m.scatter(xs, ys, color=cluster_colors[labels], alpha=0.5)
    # m.scatter(xs, ys, color="k", alpha=0.1, marker=".")
    #m.plot(poly[:,0], poly[:,1])

    """
    for label in unique_labels:
        if label == -1:
            continue
        ps = points[np.where(labels == label)]
        print len(ps)
        print ps
        hull = scipy.spatial.ConvexHull(ps[:,:2])
        pxs, pys = m(ps[hull.vertices,0], ps[hull.vertices,1])
        poly = matplotlib.patches.Polygon(zip(pxs, pys), closed=True)
        plt.gca().add_patch(poly)
        #m.plot(pxs, pys, c="k")#cluster_colors[label])
    """

    #m.quiver(xs, ys, points[:,2], points[:,3], color=cluster_colors[labels], alpha=0.5)
    plt.show()
