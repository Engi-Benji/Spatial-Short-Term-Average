import matplotlib.pyplot as plt
import numpy as np
import datetime
import gc

from filters import filter_waterfall


def get_data(tdms, channelnum=-1):
    """Takes a TDMSReader object and returns scaled data from the TDMS file associated

    Keyword arguments:
        tdms -- a TDMSreader object
    """

    #THIS HAS TO RUN OR FILE INFO NEVER POPULATES
    props = tdms.get_properties()
    n_channels = tdms.fileinfo['n_channels']

    if channelnum == -1:
        first_channel = 0
        #If you want to read to the end get the channel length minus one
        last_channel = n_channels
    else:
        first_channel = channelnum
        last_channel = channelnum
    first_time_sample = 0
    last_time_sample = tdms.channel_length - 1


    some_data = tdms.get_data(first_channel, last_channel, first_time_sample, last_time_sample)
    #print('Size of data loaded: {0}'.format(some_data.shape))

    some_data = scale(some_data, props)

    return some_data


def scale(data, props):
    """Takes in TDMS data and its properties using them to scale the data as it is compressed within the file format. Returns scaled data

    𝑠𝑡𝑟𝑎𝑖𝑛𝑟𝑎𝑡𝑒 𝑛𝑚 𝑚𝑠 = 116 𝑥 𝑖𝐷𝐴𝑆 𝑣𝑎𝑙𝑢𝑒𝑠 𝑥 𝑠𝑎𝑚𝑝𝑙𝑖𝑛𝑔 𝑓𝑟𝑒𝑞 (𝐻𝑧) / 𝑔𝑎𝑢𝑔𝑒 𝑙𝑒𝑛𝑔𝑡ℎ (𝑚)

    Keyword arguments:
        data -- numpy array containing TDMS data
        props -- properties struct from TDMS reader
    """
    data = data / 8192
    data = (116 * data * props.get('SamplingFrequency[Hz]')) / props.get('GaugeLength')
    return data


def batch_scale(data_array):
    """Takes an array of TDMS data numpy arrays and scales them. Returns a numpy array of scaled TDMS data

    Keyword arguments:
        data_array -- an array of numpy arrays containing TDMS data
    """
    new_data_array = []
    for data in data_array:
        new_data_array.append(scale(data, data.get_properties()))

    return new_data_array


def max_min_strain_rate(data):
    """Takes in a 2D numpy array of TDMS data and returns the min and max values

    Keyword arguments:
        data -- A numpy array containing TDMS data
    """
    maxVal = 0
    minVal = 0

    for time in data:
        if time.max() > maxVal:
            maxVal = time.max()

        if time.min() < minVal:
            minVal = time.min()

    return maxVal, minVal


def max_min_strain_rate_bounded(data, length, start, end):
    """I actually don't know why I made this or remember how it works

    Keyword arguments:
        data -- A numpy array containing TDMS data
        length --
        start --
        end --
    """
    maxVal = 0
    minVal = 0

    for time in data:
        for sample, location in zip(time, length):
            if start < location < end:
                if sample > maxVal:
                    maxVal = sample

                if sample < minVal:
                    minVal = sample

    return maxVal, minVal


# Make a graph from a tdms reference
def full_graph(tdms, max_strain_rate=None, min_strain_rate=None, xmin=0, xmax=0, first_time_sample=0, last_time_sample=0, save=None, folder="./figures/", highcut = -1, lowcut = -1):
    """Takes a TDMS reader object and creates a graph from the associated data, this graph can then be saved to a
    selected folder if chosen to

    Keyword arguments:
        tdms -- TDMS Reader object
        max_strain_rate -- upper bound of colour bar
        min_strain_rate -- lower bound of the colour bar
        first_time_sample -- default 0 shows all data, otherwise sets the sample within the data to start generating
        the figure
        last_time_sample -- default 0 shows all data, otherwise sets the final sample used within the data for
        the figure
        save -- None by default means no saving, any value entered will cause it to save
        folder -- File location to save figures to will by default save to ./figures/
    """

    props = tdms.get_properties()

    # where does data recording start
    zero_offset = props.get('Zero Offset (m)')
    # where does each channel sit along the cable
    channel_spacing = props.get('SpatialResolution[m]') * props.get('Fibre Length Multiplier')
    # how many channels are there
    n_channels = tdms.fileinfo['n_channels']
    # distance along the cable called depth here but hey
    depth = zero_offset + np.arange(n_channels) * channel_spacing
    fs = props.get('SamplingFrequency[Hz]')

    first_channel = 0
    # If you want to read to the end get the channel length minus one
    last_channel = n_channels

    if last_time_sample == 0:
        last_time_sample = tdms.channel_length - 1

    some_data = tdms.get_data(first_channel, last_channel, first_time_sample, last_time_sample)
    some_data = scale(some_data, props)

    if highcut != -1 or lowcut != -1:
        some_data = filter_waterfall(some_data, fs, lowcut=lowcut, highcut=highcut)

    fig1 = plt.figure()

    img1 = plt.imshow(some_data, aspect='auto', interpolation='none', extent=(
        depth[first_channel], depth[last_channel - 1], ((last_time_sample - 1) / fs), (first_time_sample / fs)),
                      vmin=min_strain_rate, vmax=max_strain_rate)
    #if xmax != 0:
    #    plt.xlim(xmin, xmax)

    plt.ylabel('Time (seconds)')
    plt.xlabel('Distance (meters)')
    if first_time_sample == 0:
        plt.title(props.get('GPSTimeStamp'))
    else:
        plt.title((props.get('GPSTimeStamp') + datetime.timedelta(seconds=first_time_sample/fs)))
    plt.set_cmap(plt.cm.get_cmap('bwr'))
    fig1.colorbar(img1, label="Nano Strain per Second [nm/m/s]")

    if save is not None:
        plt.savefig(folder + save)
        plt.clf()
        plt.close("all")
        gc.collect()
    else:
        plt.show(block=False)



if __name__ == '__main__':
    print("TESTING METHODS")
