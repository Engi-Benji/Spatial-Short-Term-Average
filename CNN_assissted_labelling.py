import os
import pickle
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt, patches
from sympy import false
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import patches

from TDMS_Batch_Reader import load_folder, sort_array
from TDMS_Utilities import get_data
from clustering import cluster_association

from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from filters import filter_waterfall, butter_bandpass_filter

import torch
from torch.utils.data import DataLoader, TensorDataset
from PyTorch_Darknet53_master.model import darknet53
from keras.utils import to_categorical


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

def CNN_window_preparation(cluster_data, data, filtered_data, window_width, window_height, offset = 0):

    raw = []
    filtered = []
    labels = []

    for event_number, cluster in enumerate(cluster_data):

        sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2) - offset
        channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)

        # bounds = 1000
        # fig1, ax = plt.subplots()
        # img1 = plt.imshow(data, cmap='bwr', vmin=-bounds, vmax=bounds)
        #
        # rect = patches.Rectangle(((channel_midp - int(window_width/2)), (sample_midp - int(window_height/2))), 80, 120, linewidth=2, edgecolor="r", facecolor='none')
        # ax.add_patch(rect)
        #
        # fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
        # plt.show()
        #
        # bounds = 1000
        # fig1 = plt.figure()
        # img1 = plt.imshow(data[(sample_midp - int(window_height/2)):(sample_midp + int(window_height/2)), (channel_midp - int(window_width/2)):(channel_midp + int(window_width/2))], cmap='bwr', vmin=-bounds, vmax=bounds)
        # fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
        # fig1.show()
        #
        # bounds = 1000
        # fig1 = plt.figure()
        # img1 = plt.imshow(filtered_data[(sample_midp - int(window_height/2)):(sample_midp + int(window_height/2)), (channel_midp - int(window_width/2)):(channel_midp + int(window_width/2))], cmap='bwr', vmin=-bounds, vmax=bounds)
        # fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
        # fig1.show()

        w = data[(sample_midp - int(window_height/2)):(sample_midp + int(window_height/2) + 1), (channel_midp - int(window_width/2)):(channel_midp + int(window_width/2) + 1)]
        fw = filtered_data[(sample_midp - int(window_height/2)):(sample_midp + int(window_height/2) + 1), (channel_midp - int(window_width/2)):(channel_midp + int(window_width/2) + 1)]

        raw.append(w)
        filtered.append(fw)
        labels.append("u")

    return [raw, filtered, labels]

def label_cluster(cnn_model_path, cluster_data, data, filtered_data, save, window_width, window_height, offset = 0):

    windows = CNN_window_preparation(cluster_data, data, filtered_data, window_width, window_height, offset)
    use_raw = True

    window_height += 1
    window_width += 1

    labelled_clusters = []

    if not len(windows[0]) == 0:

        win_raw = windows[0]
        win_filtered = windows[1]
        labels = windows[2]

        temp_raw = []
        temp_filtered = []
        temp_labels = []
        temp_clusters = []

        for i, (w, f, l, c) in enumerate(zip(win_raw, win_filtered, labels, cluster_data)):
            if np.shape(w)[0] == window_height and np.shape(w)[1] == window_width:
                temp_raw.append(w)
                temp_filtered.append(f)
                temp_labels.append(l)
                temp_clusters.append(c)

        win_raw = np.array(temp_raw)
        win_filtered = np.array(temp_filtered)
        labels = temp_labels
        clusters = temp_clusters

        del temp_raw, temp_filtered, temp_labels, temp_clusters

        win_raw_normalised = []
        win_filtered_normalised = []
        for w, f in zip(win_raw, win_filtered):

            minVal = w.min()
            maxVal = w.max()

            w_normalised = (((w - minVal) / (maxVal - minVal)) - 0.5) * 2
            win_raw_normalised.append(w_normalised)

            minVal = f.min()
            maxVal = f.max()

            f_normalised = (((f - minVal) / (maxVal - minVal)) - 0.5) * 2
            win_filtered_normalised.append(f_normalised)

        win_raw_normalised = np.array(win_raw_normalised)
        win_filtered_normalised = np.array(win_filtered_normalised)

        # stacks = []
        # for win in win_raw_normalised:
        #     stack = None
        #     for i in range(len(win[0])):
        #         if stack is None:
        #             stack = deepcopy(win[:, i])
        #         else:
        #             stack = stack + win[:, i]
        #
        #     stack = stack / len(win[0])
        #     stacks.append(stack)
        #
        # fstacks = []
        # for win in win_filtered_normalised:
        #     stack = None
        #     for i in range(len(win[0])):
        #         if stack is None:
        #             stack = deepcopy(win[:, i])
        #         else:
        #             stack = stack + win[:, i]
        #
        #     stack = stack / len(win[0])
        #     fstacks.append(stack)
        #
        # fs = 1000
        #
        # spectra = []
        # for stack in stacks:
        #     yf = rfft(stack)
        #
        #     #padding the end with 0s
        #     empty = np.zeros(window_height)
        #     for i in range(len(yf)):
        #         empty[i] = yf[i]
        #
        #     spectra.append(empty)
        #
        # features = []
        # for win, fwin, stack, fstack in zip(win_raw_normalised, win_filtered_normalised, stacks, fstacks):
        #     s_max = np.max(win)
        #     s_min = np.min(win)
        #
        #     fs_max = np.max(fwin)
        #     fs_min = np.min(fwin)
        #
        #     #temporal stacked
        #
        #     t_mean = np.mean(stack)
        #     t_std = np.std(stack)
        #     t_skew = skew(stack)
        #     t_kurtosis = kurtosis(stack)
        #
        #     ft_mean = np.mean(fstack)
        #     ft_std = np.std(fstack)
        #     ft_skew = skew(fstack)
        #     ft_kurtosis = kurtosis(fstack)
        #
        #     #spectral stacked
        #
        #     stack20 = butter_bandpass_filter(stack, 1000, 20, -1)
        #     stack100 = butter_bandpass_filter(stack, 1000, 100, -1)
        #
        #     N = len(stack)
        #     yf = rfft(stack)
        #     xf = rfftfreq(N, 1/fs)
        #
        #     N = len(stack20)
        #     yf20 = rfft(stack20)
        #     xf20 = rfftfreq(N, 1/fs)
        #
        #     N = len(stack100)
        #     yf100 = rfft(stack100)
        #     xf100 = rfftfreq(N, 1/fs)
        #
        #     sp_kurtosis = kurtosis(np.abs(yf))
        #     sp_skew = skew(np.abs(yf))
        #     sp_domFreq = xf[np.where(np.abs(yf) == np.max(np.abs(yf)))[0]][0]
        #     sp_domFreq_amp = np.max(np.abs(yf))
        #
        #     sp_domFreq_20 = xf20[np.where(np.abs(yf20) == np.max(np.abs(yf20)))[0]][0]
        #     sp_domFreq_amp_20 = np.max(np.abs(yf20))
        #
        #     sp_domFreq_100 = xf100[np.where(np.abs(yf100) == np.max(np.abs(yf100)))[0]][0]
        #     sp_domFreq_amp_100 = np.max(np.abs(yf100))
        #
        #     feature = [s_max, s_min, fs_max, fs_min, t_mean, t_std, t_skew, t_kurtosis, ft_mean, ft_std, ft_skew, ft_kurtosis, sp_kurtosis, sp_skew, sp_domFreq, sp_domFreq_amp, sp_domFreq_20, sp_domFreq_amp_20, sp_domFreq_100, sp_domFreq_amp_100]
        #
        #     # padding the end with 0s
        #     empty = np.zeros(window_height)
        #     for i in range(len(feature)):
        #         empty[i] = feature[i]
        #
        #     features.append(empty)
        #
        # modified_data = []
        # if use_raw:
        #
        #     for win, stack, fstack, spec, feature in zip(win_raw_normalised, stacks, fstacks, spectra, features):
        #         win = np.append(win, np.reshape(stack, (window_height, 1)), axis=1)
        #         win = np.append(win, np.reshape(fstack, (window_height, 1)), axis=1)
        #         win = np.append(win, np.reshape(spec, (window_height, 1)), axis=1)
        #         win = np.append(win, np.reshape(feature, (window_height, 1)), axis=1)
        #         modified_data.append(win)
        #
        # else:
        #
        #     for win, stack, fstack, spec, feature in zip(win_filtered_normalised, stacks, fstacks, spectra, features):
        #         win = np.append(win, np.reshape(stack, (window_height, 1)), axis=1)
        #         win = np.append(win, np.reshape(fstack, (window_height, 1)), axis=1)
        #         win = np.append(win, np.reshape(spec, (window_height, 1)), axis=1)
        #         win = np.append(win, np.reshape(feature, (window_height, 1)), axis=1)
        #         modified_data.append(win)
        #
        # win_modified = modified_data

        # for stack, fstack, spec, feature in zip(stacks, fstacks, spectra, features):
        #     stack = np.reshape(stack, (height, 1))
        #     stack = np.append(stack, np.reshape(fstack, (height, 1)), axis=1)
        #     stack = np.append(stack, np.reshape(spec, (height, 1)), axis=1)
        #     stack = np.append(stack, np.reshape(feature, (height, 1)), axis=1)
        #     modified_data.append(stack)
        #
        # win_modified = modified_data

        if use_raw:
            win_modified = win_raw_normalised
        else:
            win_modified = win_filtered_normalised

        # spectrograms = []
        # for stack in stacks:
        #     f, t, Sxx = signal.spectrogram(stack, fs, nperseg=12, mode= "magnitude")
        #     spectrograms.append(Sxx)
        #
        # win_raw = spectrograms

        # combo = []
        # for w, wf in zip(win_raw, win_filtered):
        #     combo.append(np.append(w, wf, axis=1))
        #
        # win_raw = combo
        #
        # bounds = 1000
        #
        # fig1 = plt.figure()
        # img1 = plt.imshow(combo[0], cmap='bwr', vmin=-bounds, vmax=bounds)
        # plt.title(labels[0])
        # fig1.show()

        # reshape for pytorch
        win_modified = np.array(win_modified).astype(np.float32)
        # reshaped_raw = win_raw.reshape(-1, 1, 242, 81)
        reshaped_modified = win_modified.reshape(-1, 1, window_height, window_width)
        # reshaped_raw = win_raw.reshape(-1, 1, 7, 10)

        labels = np.array(labels)

        for l in labels:
            if l != "f":
                labels[labels == l] = "p"

        labels = np.array(labels)
        unique = np.unique(labels)
        # print(unique)

        for i, u in enumerate(unique):
            labels[labels == u] = i

        labels_enc = to_categorical(labels, num_classes=2)

        win_modified_balanced = np.array(reshaped_modified)

        test_dataset = TensorDataset(torch.from_numpy(win_modified_balanced), torch.from_numpy(labels_enc))
        test_loader = DataLoader(test_dataset, shuffle=False)

        # print("3. Load Trained Model ----------")

        train_net = darknet53(2)
        train_net.load_state_dict(torch.load(cnn_model_path))

        # print("4. Classify and Save Windows ----------")
        train_net.eval()

        count = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = train_net(inputs)
                _, predicted = torch.max(outputs.data, 1)

                for w, p in zip(inputs, predicted):

                    if p.item() == 0:
                        #we do not label these but mark as unlabelled for parity and to make it easier to process later
                        c = clusters[count]

                        c[0] -= offset
                        c[1] -= offset

                        diffY = c[1] - c[0]
                        diffX = c[3] - c[2]

                        labelled_cluster = [diffY, diffX, "unlabelled"]
                        print(labelled_cluster)
                        labelled_clusters.append(labelled_cluster)

                        # fig1 = plt.figure()
                        # img1 = plt.imshow(w.squeeze(), cmap='bwr')
                        # #0 == foot, 1 == other
                        # plt.title(f"label = {p}")
                        # fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
                        # fig1.savefig(f"./images/foot/{save}-{count}.png")
                        # plt.close()
                        #
                        # bounds = 1000
                        # fig1 = plt.figure()
                        # img1 = plt.imshow(win_raw[count], cmap='bwr', vmin=-bounds, vmax=bounds)
                        # #0 == foot, 1 == other
                        # plt.title(f"original")
                        # fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
                        # fig1.savefig(f"./images/foot/{save}-{count}-raw.png")
                        # plt.close()
                        #
                        # bounds = 1000
                        # fig1 = plt.figure()
                        # img1 = plt.imshow(win_filtered[count], cmap='bwr', vmin=-bounds, vmax=bounds)
                        # #0 == foot, 1 == other
                        # plt.title(f"original")
                        # fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
                        # fig1.savefig(f"./images/foot/{save}-{count}-filtered.png")
                        # plt.close()

                    else:
                        #this is the data we want to label
                        # print("save location other")

                        # fig1 = plt.figure()
                        # img1 = plt.imshow(w.squeeze(), cmap='bwr')
                        # #0 == foot, 1 == other
                        # plt.title(f"label = {p}")
                        # fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
                        # fig1.savefig(f"./images/other/{save}-{count}.png")
                        # plt.close()
                        #
                        # bounds = 1000
                        # fig1 = plt.figure()
                        # img1 = plt.imshow(win_raw[count], cmap='bwr', vmin=-bounds, vmax=bounds)
                        # #0 == foot, 1 == other
                        # plt.title(f"original")
                        # fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
                        # fig1.savefig(f"./images/other/{save}-{count}-raw.png")
                        # plt.close()
                        #
                        # bounds = 1000
                        # fig1 = plt.figure()
                        # img1 = plt.imshow(win_filtered[count], cmap='bwr', vmin=-bounds, vmax=bounds)
                        # #0 == foot, 1 == other
                        # plt.title(f"original")
                        # fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
                        # fig1.savefig(f"./images/other/{save}-{count}-filtered.png")
                        # plt.close()

#-----------------------------------------------------------------------------------------------------------------------
                        c = clusters[count]
                        windowData = []
                        fWindowData = []

                        c[0] -= offset
                        c[1] -= offset

                        diffY = c[1] - c[0]
                        diffX = c[3] - c[2]

                        if diffY > 100:
                            minY = c[0] - 100
                            if minY < 0:
                                minY = 0
                            maxY = c[1] + 100
                            ylen = 100
                        else:
                            minY = c[0] - 50
                            if minY < 0:
                                minY = 0
                            maxY = c[1] + 50
                            ylen = 50

                        if diffX > 100:
                            minX = c[2] - 100
                            maxX = c[3] + 100
                            xlen = 100
                        else:
                            minX = c[2] - 50
                            maxX = c[3] + 50
                            xlen = 50


                        windowData.append(data[minY:maxY, minX: maxX])
                        fWindowData.append(filtered_data[minY:maxY, minX: maxX])

                        bounds = 1000
                        fig, ax = plt.subplots()
                        img1 = ax.imshow(data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
                        # ax.set_xlim(1200, 1500)
                        plt.ylabel('Time (seconds)')
                        img1.set_cmap(plt.cm.get_cmap('bwr'))
                        plt.show(block=False)
                        plt.close()

                        bounds = 1000
                        fig, ax = plt.subplots()
                        img1 = ax.imshow(data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
                        # ax.set_xlim(1200, 1500)
                        plt.ylabel('Time (seconds)')
                        img1.set_cmap(plt.cm.get_cmap('bwr'))

                        rect = patches.Rectangle((c[2], c[0]), c[3] - c[2], (c[1] - c[0]), linewidth=2, edgecolor="black",
                                                 facecolor='none')
                        ax.add_patch(rect)
                        plt.show(block=False)
                        plt.close()

                        bounds = 1000
                        fig, ax = plt.subplots()
                        img1 = ax.imshow(windowData[0], aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
                        plt.ylabel('Time (milliseconds)')
                        plt.title(f'Window {count}')
                        img1.set_cmap(plt.colormaps['bwr'])

                        rect = patches.Rectangle(((int(xlen - int(diffX)/2)), int(ylen - int(diffY)/2)), c[3] - c[2], (c[1] - c[0]), linewidth=2,
                                                 edgecolor="black", facecolor='none')

                        ax.add_patch(rect)

                        plt.show()
                        plt.close()

                        bounds = 1000
                        fig, ax = plt.subplots()
                        img1 = ax.imshow(fWindowData[0], aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
                        plt.ylabel('Time (milliseconds)')
                        plt.title(f'Window {count}')
                        img1.set_cmap(plt.colormaps['bwr'])

                        rect = patches.Rectangle(((int(xlen - int(diffX)/2)), int(ylen - int(diffY)/2)), c[3] - c[2], (c[1] - c[0]), linewidth=2,
                                                 edgecolor="black", facecolor='none')

                        ax.add_patch(rect)

                        plt.show()
                        plt.close()

                        bounds = 1000
                        fig, ax = plt.subplots()
                        img1 = ax.imshow(fWindowData[0], aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
                        plt.ylabel('Time (milliseconds)')
                        plt.title(f'Window {count}')
                        img1.set_cmap(plt.colormaps['bwr'])

                        plt.show()
                        plt.close()

                        labelled_cluster = [diffY, diffX]

                        print(labelled_cluster)
                        label = input("Event Type>>>")

                        labelled_cluster.append(label)
                        labelled_clusters.append(labelled_cluster)
                        print(labelled_cluster)
                        # print("i wanna break here")

                    count += 1

    # print("i wanna break here")
    with open(f"{save}", "wb") as fp:  # Pickling
        pickle.dump(labelled_clusters, fp)

def ten_sec_labelling(tdms_folder, clusters_folder, cnn_model_path, save, window_width, window_height):

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

            if file_number >= 231:
                if file_number == 0:

                    with open(f"{clusters_folder}/{clusters_array[file_number]}", "rb") as fp:   # Unpickling
                        cluster_data = pickle.load(fp)

                    cluster_data = cluster_association(cluster_data, 0.25, 40, 1000, -1, 100, -1, 1000)

                    label_cluster(cnn_model_path, cluster_data, data, filtered_data, f"{save}{filenames[file_number]}", window_width, window_height)

                else:

                    combined_data = np.append(prior_data, data, axis=0)
                    combined_filtered_data = np.append(prior_fdata, filtered_data, axis=0)

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

                    label_cluster(cnn_model_path, cluster_data, combined_data, combined_filtered_data, f"{save}{filenames[file_number]}", window_width, window_height, offset=20)

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

    device = "G:"
    window = "NDay"
    save = F"{device}/New Data/NovemberNightNew/Assisted Labelled Data/"

    tdms_folder = f"{device}/New Data/NovemberNightNew/NightNew/"
    clusters_folder = f"{device}/New Data/NovemberNightNew/Clusters/"
    cnn_model_path = f"./models/80x120_raw_CNN.pth"

    ten_sec_labelling(tdms_folder, clusters_folder, cnn_model_path, save, 80, 120)