from TDMS_Read import TdmsReader
import os
import numpy as np

def load_folder(directory, percent = 1, section = 0):
    """Loads in a folder of TDMS files into an array of TDMS reader objects, can be given a percent and section.
    Percent breaks the data down into groups of a percent size of the max number of files and section dictates which
    group is loaded

    Keyword arguments:
        directory -- The file directory containing the group of TDMS Files
        percent -- Default 1 which is 100% of the data, input a float to break data down (e.g: 0.5 will break the data
        down into sections 50% of the size of all the data)
        section -- Default 0 which is the first grouping of data can be combined with percent to access different
        groupings of data
    """
    # print(len([filename for filename in os.listdir(directory) if filename.endswith(".tdms")]))
    tdms_array = np.empty(int(len([filename for filename in os.listdir(directory) if filename.endswith(".tdms")]) * percent),
                          TdmsReader)

    length = len(tdms_array)
    lower_bound = length * section
    upper_bound = lower_bound + length

    if upper_bound > len([filename for filename in os.listdir(directory) if filename.endswith(".tdms")]):
        upper_bound = len([filename for filename in os.listdir(directory) if filename.endswith(".tdms")])
        tdms_array = np.empty(upper_bound - lower_bound, TdmsReader)

    count = 0
    for i, filename in enumerate(sorted(os.listdir(directory))):
        if filename.endswith(".tdms") and lower_bound <= i < upper_bound:
            tdms_array[count] = TdmsReader(directory + filename)
            count += 1

    return tdms_array


def sort_array(tdms_array):
    """Sorts an array of TDMS readers by GPSTimeStamp putting them in chronological order

    Keyword arguments:
        tdms_array -- An array of TDMS readers
    """
    dates = []

    for tdms in tdms_array:
        props = tdms.get_properties()
        dates.append(props.get('GPSTimeStamp'))

    sorted_array = [x for y, x in sorted(zip(np.array(dates), tdms_array))]
    return sorted_array


if __name__ == '__main__':
    print("TESTING METHODS")

    # batch loading
    array = load_folder("./30mins/")
    props = array[0].get_properties()

    zero_offset = props.get('Zero Offset (m)')
    # where does each channel sit along the cable
    channel_spacing = props.get('SpatialResolution[m]') * props.get('Fibre Length Multiplier')
    # how many channels are there
    n_channels = array[0].fileinfo['n_channels']
    # distance along the cable called depth here but hey
    depth = zero_offset + np.arange(n_channels) * channel_spacing
    # sampling frequency
    fs = props.get('SamplingFrequency[Hz]')

    print('Number of channels in file: {0}'.format(n_channels))
    print('Time samples in file: {0}'.format(array[0].channel_length))
    print('Sampling frequency (Hz): {0}'.format(fs))

    # sorting
    sorted1 = sort_array(array)
    for tdms in sorted1:
        props = tdms.get_properties()
        print(props.get('GPSTimeStamp'))
