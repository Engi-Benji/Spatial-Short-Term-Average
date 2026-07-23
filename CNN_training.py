from copy import deepcopy

from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from filters import butter_bandpass_filter

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from PyTorch_Darknet53_master.model import darknet53

def data_loading(root_folder, width, height):
    #load in the data
    folders = sorted(os.listdir(root_folder), reverse=False)

    labels = []
    win_raw = []
    win_filtered = []
    # %%
    # excluding feb currently as I have messed up the data extrcation ;)
    first = True
    count = 0
    for folder in folders:

        windows = sorted(os.listdir(f"{root_folder}/{folder}"))
        test=1

        for window in windows:
            with (open(f"{root_folder}/{folder}/{window}", "rb") as fp):  # Unpickling
                window_import = pickle.load(fp)

                if window_import[0] == "c":
                    count += 1

                labels.append(window_import[0])
                win_raw.append(np.array(window_import[1]))
                win_filtered.append(np.array(window_import[2]))

                if test == 0:
                    bounds = 1000
                    fig1 = plt.figure()
                    img1 = plt.imshow(window_import[1], cmap='bwr', vmin=-bounds, vmax=bounds)
                    #0 == foot, 1 == other
                    plt.title(f"{folder} raw")
                    fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
                    fig1.show()

                    bounds = 1000
                    fig1 = plt.figure()
                    img1 = plt.imshow(window_import[2], cmap='bwr', vmin=-bounds, vmax=bounds)
                    #0 == foot, 1 == other
                    plt.title(f"{folder} filtered")
                    fig1.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")
                    fig1.show()
                    test = 1



    #remove irregular sized file
    count = 0
    temp_raw = []
    temp_filtered = []
    temp_labels = []

    for i, ((w, f), l) in enumerate(zip(zip(win_raw, win_filtered), labels)):
        if np.shape(w)[0] == height and np.shape(w)[1] == width:
            temp_raw.append(w)
            temp_filtered.append(f)
            temp_labels.append(l)
            count += 1

    win_raw = np.array(temp_raw)
    win_filtered = np.array(temp_filtered)
    labels = temp_labels

    print(f"Total Samples: {len(win_raw)}")
    return win_raw, win_filtered, labels

if __name__ == '__main__':
    print("------------ Data Preparation ------------")
    print('     1. Loading Data')

    num_epochs = 30
    width = 80
    height = 120
    use_raw = False

    root_folder = f"G:/CNN Formatted Data - {width} x {height}"
    model_save = f"./models/{width}x{height}_{'raw' if use_raw else 'filtered'}_filtered_CNN.pth"

    height += 1
    width += 1

    win_raw, win_filtered, labels = data_loading(root_folder, width, height)

    stacks = []
    for win in win_raw:
        stack = None
        for i in range(len(win[0])):
            if stack is None:
                stack = deepcopy(win[:, i])
            else:
                stack = stack + win[:, i]

        stack = stack / len(win[0])
        stacks.append(stack)

    fstacks = []
    for win in win_filtered:
        stack = None
        for i in range(len(win[0])):
            if stack is None:
                stack = deepcopy(win[:, i])
            else:
                stack = stack + win[:, i]

        stack = stack / len(win[0])
        fstacks.append(stack)

    fs = 1000

    spectra = []
    for stack in stacks:
        yf = rfft(stack)

        #padding the end with 0s
        empty = np.zeros(height)
        for i in range(len(yf)):
            empty[i] = yf[i]

        spectra.append(empty)

    features = []
    for win, fwin, stack, fstack in zip(win_raw, win_filtered, stacks, fstacks):
        s_max = np.max(win)
        s_min = np.min(win)

        fs_max = np.max(fwin)
        fs_min = np.min(fwin)

        #temporal stacked

        t_mean = np.mean(stack)
        t_std = np.std(stack)
        t_skew = skew(stack)
        t_kurtosis = kurtosis(stack)

        ft_mean = np.mean(fstack)
        ft_std = np.std(fstack)
        ft_skew = skew(fstack)
        ft_kurtosis = kurtosis(fstack)

        #spectral stacked

        stack20 = butter_bandpass_filter(stack, 1000, 20, -1)
        stack100 = butter_bandpass_filter(stack, 1000, 100, -1)

        N = len(stack)
        yf = rfft(stack)
        xf = rfftfreq(N, 1/fs)

        N = len(stack20)
        yf20 = rfft(stack20)
        xf20 = rfftfreq(N, 1/fs)

        N = len(stack100)
        yf100 = rfft(stack100)
        xf100 = rfftfreq(N, 1/fs)

        sp_kurtosis = kurtosis(np.abs(yf))
        sp_skew = skew(np.abs(yf))
        sp_domFreq = xf[np.where(np.abs(yf) == np.max(np.abs(yf)))[0]][0]
        sp_domFreq_amp = np.max(np.abs(yf))

        sp_domFreq_20 = xf20[np.where(np.abs(yf20) == np.max(np.abs(yf20)))[0]][0]
        sp_domFreq_amp_20 = np.max(np.abs(yf20))

        sp_domFreq_100 = xf100[np.where(np.abs(yf100) == np.max(np.abs(yf100)))[0]][0]
        sp_domFreq_amp_100 = np.max(np.abs(yf100))

        feature = [s_max, s_min, fs_max, fs_min, t_mean, t_std, t_skew, t_kurtosis, ft_mean, ft_std, ft_skew, ft_kurtosis, sp_kurtosis, sp_skew, sp_domFreq, sp_domFreq_amp, sp_domFreq_20, sp_domFreq_amp_20, sp_domFreq_100, sp_domFreq_amp_100]

        # padding the end with 0s
        empty = np.zeros(height)
        for i in range(len(feature)):
            empty[i] = feature[i]

        features.append(empty)

    modified_data = []
    # if use_raw:
    #
    #     for win, stack, fstack, spec, feature in zip(win_raw, stacks, fstacks, spectra, features):
    #         win = np.append(win, np.reshape(stack, (height, 1)), axis=1)
    #         win = np.append(win, np.reshape(fstack, (height, 1)), axis=1)
    #         win = np.append(win, np.reshape(spec, (height, 1)), axis=1)
    #         win = np.append(win, np.reshape(feature, (height, 1)), axis=1)
    #         modified_data.append(win)
    #
    # else:
    #
    #     for win, stack, fstack, spec, feature in zip(win_filtered, stacks, fstacks, spectra, features):
    #         win = np.append(win, np.reshape(stack, (height, 1)), axis=1)
    #         win = np.append(win, np.reshape(fstack, (height, 1)), axis=1)
    #         win = np.append(win, np.reshape(spec, (height, 1)), axis=1)
    #         win = np.append(win, np.reshape(feature, (height, 1)), axis=1)
    #         modified_data.append(win)
    #
    # win_modified = modified_data

    for stack, fstack, spec, feature in zip(stacks, fstacks, spectra, features):
        stack = np.reshape(stack, (height, 1))
        stack = np.append(stack, np.reshape(fstack, (height, 1)), axis=1)
        stack = np.append(stack, np.reshape(spec, (height, 1)), axis=1)
        stack = np.append(stack, np.reshape(feature, (height, 1)), axis=1)
        modified_data.append(stack)

    win_modified = modified_data

    # if use_raw:
    #     win_modified = win_raw
    # else:
    #     win_modified = win_filtered

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
    print('     2. Reshaping Data')
    win_modified = np.array(win_modified).astype(np.float32)
    # reshaped_raw = win_raw.reshape(-1, 1, 242, 81)
    reshaped_modified = win_modified.reshape(-1, 1, height, 4)
    # reshaped_raw = win_raw.reshape(-1, 1, 7, 10)

    labels = np.array(labels)

    for l in labels:
        if l != "f":
            labels[labels == l] = "p"

    labels = np.array(labels)
    unique = np.unique(labels)
    print(unique)

    for i, u in enumerate(unique):
        labels[labels == u] = i

    labels_enc = to_categorical(labels, num_classes=2)

    # balancing

    print('     3. Balancing Data')
    win_modified_balanced = []
    labels_balanced = []

    countOther = 0
    for w, l in zip(reshaped_modified, labels_enc):
        if l[0] != 1:
            labels_balanced.append(l)
            win_modified_balanced.append(w)
            countOther += 1

    countFoot = 0
    for w, l in zip(reshaped_modified, labels_enc):
        if l[0] == 1:
            if countFoot != countOther:
                labels_balanced.append(l)
                win_modified_balanced.append(w)
                countFoot += 1

    win_modified_balanced = np.array(win_modified_balanced)
    labels_balanced = np.array(labels_balanced)


    # ------------- normalisation ---------------
    print('     4. Normalizing Data')
    minVal = win_modified_balanced.min()
    maxVal = win_modified_balanced.max()

    win_raw_balanced_normalised = (((win_modified_balanced - minVal) / (maxVal - minVal)) - 0.5) * 2

    print(f'Max: {win_modified_balanced.max()}, Min: {win_modified_balanced.min()}')

    # train test split
    # ================
    print('     5. Splitting Data')
    # random seed
    random_seed = 2

    # train validation split
    win_raw_train, temp_win, labels_train_enc, temp_label = train_test_split(win_raw_balanced_normalised, labels_balanced,
                                                                                 test_size=0.3)
    win_raw_val, win_raw_test, labels_val_enc, labels_test_enc = train_test_split(temp_win, temp_label,
                                                                                 test_size=0.5)

    print(f"Train: {len(win_raw_train)}")
    print(f"Validation: {len(win_raw_val)}")
    print(f"Test: {len(win_raw_test)}")

    train_dataset = TensorDataset(torch.from_numpy(win_raw_train), torch.from_numpy(labels_train_enc))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    val_dataset = TensorDataset(torch.from_numpy(win_raw_val), torch.from_numpy(labels_val_enc))
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=16)

    test_dataset = TensorDataset(torch.from_numpy(win_raw_test), torch.from_numpy(labels_test_enc))
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=16)

    #create nn and training
    print("------------ Neural Network Training ------------")

    net = darknet53(2)
    # net = SimpleNN()

    if torch.cuda.is_available():
        net = net.cuda()

    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    min_valid_loss = np.inf

    for e in range(num_epochs):
        train_loss = 0.0
        net.train()  # Optional when not using Model Specific layer
        for data, labels in train_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            optimizer.zero_grad()

            target = net(data)
            loss = loss_function(target, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        net.eval()  # Optional when not using Model Specific layer
        with torch.no_grad():
            for data, labels in val_loader:
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                target = net(data)
                loss = loss_function(target, labels)
                valid_loss = loss.item() * data.size(0)

        print(f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), model_save)

    print("------------ Neural Network Testing ------------")

    train_net = darknet53(2)
    # train_net = SimpleNN()
    train_net.load_state_dict(torch.load(model_save))

    # train_net = net

    correct = 0
    total = 0

    footTrue = 0
    footFalse = 0
    otherTrue = 0
    otherFalse = 0

    train_net.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = train_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            for l, p in zip(labels, predicted):
                if l[p] == 1:
                    correct += 1
                    if p == 0:
                        footTrue += 1
                    else:
                        otherTrue += 1
                else:
                    if p == 0:
                        footFalse += 1
                    else:
                        otherFalse += 1

    accuracy = 100 * correct / total

    print(f"  Test Accuracy: {accuracy}%")
    print(" |==f==|==0==|")
    print(f"f| {footTrue} | {footFalse} |")
    print(" |=====|=====|")
    print(f"0| {otherFalse} | {otherTrue} |")
    print(" |=====|=====|")


    # multi_class(win_raw, labels, 161, 241)
    # binary_class(win_raw, labels, 161, 241)
    # balanced_binary_class(win_raw, labels, 161, 241)
    #
    # multi_class(win_filtered, labels, 161, 241)
    # binary_class(win_filtered, labels, 161, 241)
    # balanced_binary_class(win_filtered, labels, 161, 241)

    # multi_class(combo, labels, 322, 241)
    # binary_class(combo, labels, 322, 241)
    # balanced_binary_class(combo, labels, 322, 241)