from scipy.signal import butter, sosfilt
import numpy as np
from numpy.typing import NDArray

type FloatSample = np.float16 | np.float32 | np.float64
type SOSCoefficients = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def butter_bandpass(
    lowcut: float, highcut: float, fs: float, order: int
) -> SOSCoefficients:
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
        sos = butter(order, high, analog=False, btype="low", output="sos")
    elif highcut == -1:
        sos = butter(order, low, analog=False, btype="high", output="sos")
    else:
        sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def filter_waterfall[SampleT: FloatSample](
    some_data: NDArray[SampleT],
    fs: float,
    lowcut: float = -1,
    highcut: float = -1,
    order: int = 6,
) -> NDArray[SampleT]:
    """Applies a Butterworth filter to a full 2D TDMS np array

    Keyword arguments:
        some_data -- TDMS np array
        fs -- frequency of recording
        lowcut -- lower bound for filter defaults to -1 meaning no lower bound
        highcut -- higher bound for filter defaults to -1 meaning no higher bound
        order -- the order of the filter
    """
    filtered_data = np.empty(some_data.shape, dtype=some_data.dtype)
    sos = butter_bandpass(lowcut, highcut, fs, order)

    for samp_num in range(0, len(some_data[0])):
        time_sample = some_data.transpose()[samp_num, :]

        filtered_signal = sosfilt(sos, time_sample)
        
        filtered_data[:, samp_num] = filtered_signal

    return filtered_data
