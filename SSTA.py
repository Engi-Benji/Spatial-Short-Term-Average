import gc
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from filters import filter_waterfall
from TDMS_Read import TdmsReader
from TDMS_Utilities import get_data


def plot_mask(data, x, save=None):
    """Plots an array of DAS data and the mask produced by the SSTA method

    Keyword arguments:
        data -- A 2D array of data pulled from a DAS tdms array.
        x -- A 2D array containing the mask to plot on the graph
        save -- A string for the directory to save the graph produced, if left blank the graph will not be saved
    """

    bounds = 1000

    fig, ax = plt.subplots()
    plt.set_cmap(plt.colormaps.get_cmap('bwr'))
    img1 = ax.imshow(data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
    plt.ylabel('Time (seconds)')
    plt.xlabel('Distance (Channels)')
    plt.title('2023/11/09 13:49:47 - Unprocessed')
    # ax.plot(x[1], x[0], color='black', linewidth = 1, marker='o', markerfacecolor='red', markersize=1, linestyle='None', label="Anomaly")

    bounds = 1000

    fig, ax = plt.subplots()
    plt.set_cmap(plt.colormaps.get_cmap('bwr'))
    img1 = ax.imshow(data, aspect='auto', interpolation='none', vmin=-bounds, vmax=bounds)
    plt.ylabel('Time (seconds)')
    plt.xlabel('Distance (Channels)')
    plt.title('2023/11/09 13:57:07 - Filtered and Processed')
    ax.plot(x[1], x[0], color='black', linewidth=1, marker='o', markerfacecolor='black', markersize=1, linestyle='None',
            label="Anomaly")

    if save is not None:
        plt.savefig(save)
        plt.clf()
        plt.close("all")
        gc.collect()
    else:
        plt.show(block=False)


def ssta(data, start_channel=1650, end_channel=8500, num=50, thresh=2, get_mask=True):
    """Performs the spatial short term average on an array of DAS data

    Keyword arguments:
        data -- A 2D array of DAS data
        start_channel -- An int. First column (channel) of data the method will use
        end_channel -- An int. Last column (channel of data the method will use)
        num -- An int. Number of rows (time samples) used to perform the SSTA
        thresh -- An int. The threshold ratio above which indicates an event has occured
        get_mask -- A boolean flag indicating the returned arrays format:
            True - -1 (Not processed), 0 (No Event), 1 (Event)
            False - -1 (Not processed), ratio value
    """

    n_channels = data.shape[1]
    mask = []
    with tqdm(total=len(data)) as pbar:
        for i in range(0, len(data)):
            if not (i - num <= 0):
                temp = np.array(data[i - num:i, start_channel:end_channel])
                temp = abs(temp)
                mean = temp.mean()

                channel_mask = []
                for channel in range(len(temp[0])):
                    cdata = np.array(temp.transpose()[channel, :])
                    cdata = abs(cdata)
                    if cdata.mean() / mean > thresh:

                        if get_mask:
                            # 1 or 0 mask
                            channel_mask.append(1)
                        else:
                            # gives actual amplitude of the data
                            channel_mask.append(abs(data[i, channel]))
                    else:
                        channel_mask.append(0)

                mask.append(channel_mask)
            pbar.update()

    mask = np.array(mask)
    mask = np.pad(mask, [(num, 0), (start_channel, (n_channels - end_channel))], mode='constant', constant_values=-1)

    return mask

if __name__ == '__main__':
    file_path = "Example Windows/November_Window_UTC_20231109_134947.573.tdms"

    tdms = TdmsReader(file_path)
    data = get_data(tdms)
    print("Data Loaded")

    highcut = -1
    lowcut = 100
    filtered_data = filter_waterfall(data, 1000, lowcut=lowcut, highcut=highcut)
    print("Data Filtered")

    mask = ssta(filtered_data, thresh=8, num=2, get_mask=False)
    x = np.where(mask > 0)

    plot_mask(data, x)
