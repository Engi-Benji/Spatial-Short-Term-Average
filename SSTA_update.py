from numpy.typing import NDArray
from tqdm import tqdm
import numpy as np


def ssta(
    data: NDArray[np.floating],
    start_channel: int = 1650,
    end_channel: int = 8500,
    num: int = 50,
    thresh: int = 2,
    get_mask: bool = True,
) -> NDArray[np.floating | np.int64]:
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
    mask: list[NDArray[np.int64] | NDArray[np.floating]] = []
    with tqdm(total=len(data)) as pbar:
        for i in range(0, len(data)):
            if not (i - num <= 0):
                temp = np.abs(data[i - num : i, start_channel:end_channel])
                mean = temp.mean()

                channel_means = np.ascontiguousarray(temp.T).mean(axis=1)
                hits = channel_means / mean > thresh

                if get_mask:
                    mask.append(hits.astype(np.int64))
                else:
                    amplitudes = np.abs(data[i, : hits.size])

                    if not hits.any():
                        mask.append(np.zeros(hits.size, np.int64))
                    elif hits.all():
                        mask.append(amplitudes)
                    else:
                        mask.append(
                            np.where(hits, amplitudes, 0).astype(
                                np.result_type(data.dtype, np.int64)
                            )
                        )

            pbar.update()

    return np.pad(
        np.array(mask),
        [(num, 0), (start_channel, (n_channels - end_channel))],
        mode="constant",
        constant_values=-1,
    )
