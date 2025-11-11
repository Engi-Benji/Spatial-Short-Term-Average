import gc
from sklearn.cluster import HDBSCAN
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt

from TDMS_Read import TdmsReader
from TDMS_Utilities import get_data
from filters import filter_waterfall
from SSTA import ssta


def plot_clusters(data, cluster_info=None, x=None, min_width=-1, max_width=-1, num_points=-1, save=None):
    """Plots

    Keyword arguments:
        data -- A 2D array containing DAS data
        cluster_info -- A 2D array containing the properties of the clusters within the parsed data
        x -- A 2D array containing the locations of event points within the window
        min_width -- An int indicating the minimum width a cluster must have to be shown.
        -1 indicates there is no minimum width
        max_width -- An int indicating the maximum width a cluster must have to be shown.
        -1 indicates there is no maximum width
        num_points -- An int indicating the minimum number of points a cluster must be made of.
        -1 indicates there is no minimum number of points
        save -- A string indicating where the plot generated should be saved.
        No string indicated the plot should be shown and not saved.
    """

    bounds = 1000
    fig, ax = plt.subplots()
    img1 = ax.imshow(data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
    # ax.set_xlim(1200, 1500)
    plt.ylabel('Time (seconds)')
    img1.set_cmap(plt.cm.get_cmap('bwr'))

    if cluster_info is not None:
        rects = []
        for c in cluster_info:

            # filter out the different sizes of clusters based on what we know about our event
            # spacial presence
            if c[3] - c[2] > max_width != -1:
                continue

            if c[3] - c[2] < min_width != -1:
                continue

            # Number of points flagged
            if c[4] < num_points != -1:
                continue

            rect = patches.Rectangle((c[2], c[0]), c[3] - c[2], (c[1] - c[0]), linewidth=2, edgecolor="r", facecolor='none')
            #print(f"Width: {c[3] - c[2]}, Height: {c[1] - c[0]}, Num of Points: {c[4]}")
            rects.append(rect)

        for rect in rects:
            ax.add_patch(rect)

    if x is not None:
        ax.plot(x[1], x[0], color='black', linewidth=0.5, marker='o', markerfacecolor='red', markersize=0.5,
                linestyle='None', label="Anomaly")

    if save is not None:
        plt.savefig(save)
        plt.clf()
        plt.close("all")
        gc.collect()
    else:
        plt.show(block=False)


def extract_clusters(x):
    """Method takes in an input of a set of points and clusters them using HDBScan. The resulting clusters are then
    processed to get information about their width height and the number of points that make up the cluster.

    Keyword arguments:
        x -- An 2D array containing a set of points to be clustered
    """

    point_data = list(zip(x[1], x[0]))

    hdb = HDBSCAN(min_cluster_size= 4, min_samples=None)
    clusters = hdb.fit_predict(point_data)

    #extract cluster window information from the point dataset
    cluster_vals = []
    for i in range(0, (max(clusters) + 1)):
        cluster_vals.append([])
    tx = np.transpose(x)

    for point, l in zip(tx, hdb.labels_):
        #print(point, l)
        cluster_vals[l].append(point)

    #cluster_vals[0] = np.transpose(cluster_vals[0])
    #cluster_vals[1] = np.transpose(cluster_vals[1])
    #print(cluster_vals)
    cluster_info = []
    for c in cluster_vals:
        c = np.transpose(c)
        info = []

        #time
        info.append(np.min(c[0]))
        info.append(np.max(c[0]))

        #channel
        info.append(np.min(c[1]))
        info.append(np.max(c[1]))
        info.append(np.shape(c)[1])

        cluster_info.append(info)

    return cluster_info

if __name__ == '__main__':
    file_path = "Example Windows/November_Window_UTC_20231109_134947.573.tdms"

    tdms = TdmsReader(file_path)
    data = get_data(tdms)

    highcut = -1
    lowcut = 100
    filtered_data = filter_waterfall(data, 1000, lowcut=lowcut, highcut=highcut)

    mask = ssta(filtered_data, thresh=8, num=2)
    x = np.where(mask == 1)

    cluster_info = extract_clusters(x)

    plot_clusters(data, cluster_info, x)
    plot_clusters(data, cluster_info, x, max_width=100, num_points=100)
