from scipy.signal import butter, sosfilt, sosfreqz
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order):
    """Butterworth Bandpass Filter Creator, returns a bandpass filter

    Keyword arguments:
        lowcut -- lower bound of the filter -1 for no lower bound  (lowpass)
        highcut -- higher bound of the filter -1 for no upper bound (highpass)
        fs -- frequency of recorded data
        order -- the order of the filter
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if lowcut == -1:
        sos = butter(order, high, analog=False, btype='low', output='sos')
    elif highcut == -1:
        sos = butter(order, low, analog=False, btype='high', output='sos')
    else:
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, fs, lowcut=-1, highcut=-1, order=6):
    """Applies a butterworth filter to a channel of data and returns the filtered data

    Keyword arguments:
        data -- single channel of data for filtering
        fs -- frequency of recording
        lowcut -- lower bound for filter defaults to -1 meaning no lower bound
        highcut -- higher bound for filter defaults to -1 meaning no higher bound
        order -- the order of the filter
    """
    sos = butter_bandpass(lowcut, highcut, fs, order)
    y = sosfilt(sos, data)
    return y


def filter_waterfall(some_data, fs, lowcut=-1, highcut=-1, order=6):
    """Applies a Butterworth filter to a full 2D TDMS np array

    Keyword arguments:
        some_data -- TDMS np array
        fs -- frequency of recording
        lowcut -- lower bound for filter defaults to -1 meaning no lower bound
        highcut -- higher bound for filter defaults to -1 meaning no higher bound
        order -- the order of the filter
    """
    filtered_data = np.empty(some_data.shape, type(some_data[0, 0]))

    for samp_num in range(0, len(some_data[0])):
        # print(samp_num)
        time_sample = some_data.transpose()[samp_num, :]

        filtered_signal = butter_bandpass_filter(time_sample, lowcut=lowcut, highcut=highcut, fs=fs, order=order)

        for i, point in enumerate(filtered_signal):
            filtered_data[i, samp_num] = point

    return filtered_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = sosfreqz(sos, worN=2000)
        plt.plot(w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.arange(0, nsamples) / fs
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()
