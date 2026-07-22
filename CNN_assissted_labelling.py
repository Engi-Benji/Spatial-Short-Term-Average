import os
import pickle
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt, patches
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import patches

from TDMS_Batch_Reader import load_folder, sort_array
from TDMS_Utilities import get_data
from clustering import cluster_association
from filters import filter_waterfall

def test_graphs(data, col, fig = None, ax = None):
    if fig is None:
        fig, ax = plt.subplots()

    rects = []
    count1 = 0
    for c in data:

        # filter out the different sizes of clusters based on what we know about our event
        # spacial presence
        if c[3] - c[2] > 100 != -1:
            continue

        if c[3] - c[2] < 20 != -1:
            continue

        rect = patches.Rectangle((c[2], c[0]), c[3] - c[2], (c[1] - c[0]), linewidth=2, edgecolor=col, facecolor='none')
        # print(f"Width: {c[3] - c[2]}, Height: {c[1] - c[0]}, Num of Points: {c[4]}")
        rects.append(rect)
        count1 += 1

    for rect in rects:
        ax.add_patch(rect)

    return fig, ax

def label_cluster(cluster_data, data, filtered_data, save, window_width, window_height, offset = 0):

    for event_number, cluster in enumerate(cluster_data):

        sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2) - offset
        channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)

        bounds = 1000
        fig1, ax = plt.subplots()
        img1 = plt.imshow(data, cmap='bwr', vmin=-bounds, vmax=bounds)

        rect = patches.Rectangle(((channel_midp - int(window_width/2)), (sample_midp - int(window_height/2))), 80, 120, linewidth=2, edgecolor="r", facecolor='none')
        ax.add_patch(rect)

        fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
        plt.show()

        bounds = 1000
        fig1 = plt.figure()
        img1 = plt.imshow(data[(sample_midp - int(window_height/2)):(sample_midp + int(window_height/2)), (channel_midp - int(window_width/2)):(channel_midp + int(window_width/2))], cmap='bwr', vmin=-bounds, vmax=bounds)
        fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
        fig1.show()

        bounds = 1000
        fig1 = plt.figure()
        img1 = plt.imshow(filtered_data, cmap='bwr', vmin=-bounds, vmax=bounds)
        fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
        plt.savefig(f"{save}-{event_number}.png", dpi=300)

        bounds = 1000
        fig1 = plt.figure()
        img1 = plt.imshow(filtered_data[(sample_midp - int(window_height/2)):(sample_midp + int(window_height/2)), (channel_midp - int(window_width/2)):(channel_midp + int(window_width/2))], cmap='bwr', vmin=-bounds, vmax=bounds)
        fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
        fig1.show()

        w = data[(sample_midp - int(window_height/2)):(sample_midp + int(window_height/2)), (channel_midp - int(window_width/2)):(channel_midp + int(window_width/2))]
        fw = filtered_data[(sample_midp - int(window_height/2)):(sample_midp + int(window_height/2)), (channel_midp - int(window_width/2)):(channel_midp + int(window_width/2))]


def tenseclabelling(tdms_folder, clusters_folder, save, window_width, window_height):

    filenames = sorted([filename for filename in os.listdir(tdms_folder)])

    tdms_array = load_folder(tdms_folder)
    tdms_array = sort_array(tdms_array)
    clusters_array = sorted(os.listdir(clusters_folder))

    prior_data = None
    prior_fdata = None

    with tqdm(total=len(tdms_array)) as pbar:
        for file_number, tdms in enumerate(tdms_array):

            data = get_data(tdms)
            filtered_data = filter_waterfall(data, 1000, 100)

            if file_number == 0:

                with open(f"{clusters_folder}/{clusters_array[file_number]}", "rb") as fp:   # Unpickling
                    cluster_data = pickle.load(fp)

                cluster_data = cluster_association(cluster_data, 0.25, 40, 1000, -1, 100, -1, 1000)

                label_cluster(cluster_data, data, filtered_data, f"{save}-{file_number}", window_width, window_height)

            else:

                combined_data = np.append(prior_data, data, axis=0)
                filtered_data = np.append(prior_fdata, filtered_data, axis=0)

                clust_num = ((file_number + 1) * 2) - 1

                with open(f"{clusters_folder}/{clusters_array[clust_num - 3]}", "rb") as fp:   # Unpickling
                    prior_cluster_data = pickle.load(fp)

                with open(f"{clusters_folder}/{clusters_array[clust_num - 2]}", "rb") as fp:   # Unpickling
                    overlap_cluster_data = pickle.load(fp)

                with open(f"{clusters_folder}/{clusters_array[clust_num - 1]}", "rb") as fp:   # Unpickling
                    cluster_data = pickle.load(fp)


                combined = []
                for c in cluster_data:
                    combined.append([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]])

                combined.extend(x for x in overlap_cluster_data if x not in combined)
                combined = (x for x in combined if x not in prior_cluster_data)

                cluster_data = combined
                del combined

                temp = []
                for c in cluster_data:
                    if 20 <= c[3] - c[2] < 100 and c[1] != c[0] > 0:
                        temp.append(c)

                cluster_data = temp
                del temp

                cluster_data = cluster_association(cluster_data, 0.25, 40, 1000, -1, 100, -1, 1000)

                label_cluster(cluster_data, combined_data, filtered_data, f"{save}-{file_number}", window_width, window_height, offset=20)

            prior_data = data
            prior_fdata = filtered_data
            pbar.update(1)


# def thirtyseclabelling():
#
#     device = "G:/"
#
#     directory = f"{device}Clusters/FNight/"
#     cluster_array = sorted([filename for filename in os.listdir(directory)])
#     #cluster_array = cluster_array[0:5]
#
#     tdms_array = load_folder(f"{device}1000Hz Data/FNight/")
#     tdms_array = sort_array(tdms_array)
#     #tdms_array = tdms_array[0:3]
#
#     highcut = -1
#     lowcut = 100
#
#     priordata = None
#     prior_fdata = None
#     prior_cInfo = None
#
#     count = 1
#     total = len(tdms_array)
#     for i, tdms in enumerate(tdms_array):
#         print(f"-----------------{count}/{total}-----------------")
#
#
#         data = getData(tdms)
#         filtered_data = filter_waterfall(data, 1000, lowcut=lowcut, highcut=highcut)
#
#         future_prior = []
#         combined = []
#         #3
#         if count >= 1:
#             if priordata is None:
#
#                 for j in range(5):
#                     with open(f"{directory}{cluster_array[i+j]}", "rb") as fp:  # Unpickling
#                         cInfo = pickle.load(fp)
#
#                     if j == 0:
#                         for c in cInfo:
#                             combined.append([c[0], c[1], c[2], c[3], c[4]])
#                     elif j == 4:
#                         combined.extend(x for x in cInfo if x not in combined)
#                         for c in cInfo:
#                             future_prior.append([c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]])
#                     else:
#                         combined.extend(x for x in cInfo if x not in combined)
#
#                 # fig, ax = test_graphs(combined, "g")
#                 # plt.xlim(0, 9000)
#                 # plt.ylim(60000, 0)
#                 # plt.show()
#                 # plt.close(fig)
#
#                 combined = unlabeled_association(combined, 0.25, 1000)
#                 label_cinfo(combined, data, filtered_data,
#                             f"{device}Labeled Clusters and Features/FDay/{cluster_array[2*i]}",
#                             f"{device}Labeled Clusters and Features/FDay/{cluster_array[2*i]}",
#                             low = 20)
#
#                 # fig, ax = test_graphs(combined, "r")
#                 # plt.xlim(0, 9000)
#                 # plt.ylim(60000, 0)
#                 # plt.show()
#                 # plt.close(fig)
#                 #
#                 # bounds = 1000
#                 # fig, ax = plt.subplots()
#                 # img1 = ax.imshow(data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
#                 # # ax.set_xlim(1200, 1500)
#                 # plt.ylabel('Time (seconds)')
#                 # img1.set_cmap(plt.cm.get_cmap('bwr'))
#                 # plt.show(block=False)
#                 # plt.close()
#                 #
#                 # bounds = 1000
#                 # fig, ax = plt.subplots()
#                 # img1 = ax.imshow(filtered_data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
#                 # # ax.set_xlim(1200, 1500)
#                 # plt.ylabel('Time (seconds)')
#                 # img1.set_cmap(plt.cm.get_cmap('bwr'))
#                 # plt.show(block=False)
#                 # plt.close()
#
#
#
#             else:
#                 #data = getData(tdms)
#                 combined_data = np.append(priordata, data, axis=0)
#                 #filtered_data = filter_waterfall(data, 1000, lowcut=lowcut, highcut=highcut)
#                 combined_fdata = np.append(prior_fdata, filtered_data, axis=0)
#
#                 clust_num = ((i + 1) * 6) - 1
#
#                 combined = []
#                 for j in range(1, 6):
#                     with open(f"{directory}{cluster_array[clust_num - j]}", "rb") as fp:  # Unpickling
#                         cInfo = pickle.load(fp)
#
#                     if j == 0:
#                         for c in cInfo:
#                             combined.append([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]])
#                     if j == 1:
#                         for c in cInfo:
#                             combined.extend([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] for c in cInfo if [c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] not in combined)
#                             future_prior.append([c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]])
#                     else:
#                         combined.extend([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] for c in cInfo if [c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] not in combined)
#
#                 with open(f"{directory}{cluster_array[clust_num - 6]}", "rb") as fp:  # Unpickling
#                     cInfo = pickle.load(fp)
#
#                 combined.extend([c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]] for c in cInfo if [c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]] not in combined)
#                 combined = (x for x in combined if x not in prior_cInfo)
#
#
#
#                 # fig, ax = test_graphs(prior_cInfo, "b")
#                 # fig, ax = test_graphs(combined, "g", fig, ax)
#                 # plt.xlim(0, 9000)
#                 # plt.ylim(60000, 0)
#                 # plt.show()
#                 # plt.close(fig)
#
#                 combined = unlabeled_association(combined, 0.25, 1000)
#                 label_cinfo(combined, combined_data, combined_fdata,
#                             f"{device}Labeled Clusters and Features/FDay/{cluster_array[6*i]}",
#                             f"{device}Labeled Clusters and Features/FDay/{cluster_array[6*i]}",
#                             low = 20, offset = 20)
#
#                 # fig, ax = test_graphs(combined, "r")
#                 # plt.xlim(0, 9000)
#                 # plt.ylim(60000, 0)
#                 # plt.show()
#                 # plt.close(fig)
#                 #
#                 # bounds = 1000
#                 # fig, ax = plt.subplots()
#                 # img1 = ax.imshow(combined_data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
#                 # # ax.set_xlim(1200, 1500)
#                 # plt.ylabel('Time (seconds)')
#                 # img1.set_cmap(plt.cm.get_cmap('bwr'))
#                 # plt.show(block=False)
#                 # plt.close()
#                 #
#                 # bounds = 1000
#                 # fig, ax = plt.subplots()
#                 # img1 = ax.imshow(combined_fdata, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
#                 # # ax.set_xlim(1200, 1500)
#                 # plt.ylabel('Time (seconds)')
#                 # img1.set_cmap(plt.cm.get_cmap('bwr'))
#                 # plt.show(block=False)
#                 # plt.close()
#
#
#         priordata = data[19999::]
#         prior_fdata = filtered_data[19999::]
#         prior_cInfo = future_prior
#         count += 1

if __name__ == '__main__':

    device = "G"
    window = "NDay"
    save = F"testing"

    tdms_folder = f"{device}:/1000Hz Data/{window}/"
    clusters_folder = f"{device}:/Clusters/{window}/"

    tenseclabelling(tdms_folder, clusters_folder, save, 80, 120)