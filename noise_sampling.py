import numpy as np
import random
import pickle
from filters import filter_waterfall
import matplotlib.pyplot as plt
from TDMS_Batch_Reader import load_folder
from TDMS_Utilities import get_data


def noise_sampling(tdms_directory, cluster_directory, label_directory, save):

    tdms_array = load_folder(f"{tdms_directory}/")

    highcut = -1
    lowcut = 100

    sample_count = 0
    for tdms in tdms_array:
        props = tdms.get_properties()
        time = str(props.get('GPSTimeStamp')).replace(':', "-")

        data = get_data(tdms)
        fdata = filter_waterfall(data, 1000, lowcut=lowcut, highcut=highcut)

        (height, width) = np.shape(data)

        print(width, height)

        clusters = []
        labels = []

        event_count = 0
        while event_count < 3:
            x = random.randrange(0, width)
            y = random.randrange(0, height)

            print(x, y)

            bounds = 1000


            fig, ax = plt.subplots()

            img1 = ax.imshow(data, vmin=-bounds, vmax=bounds)
            img1.set_cmap(plt.colormaps['bwr'])

            plt.xlim(x - 40, x + 40)
            plt.ylim(y + 60, y - 60)

            plt.ylabel('Time (seconds)')
            plt.xlabel('Distance (Channels)')

            fig.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")

            plt.show(block=False)
            plt.close()

            bounds = 1000


            fig, ax = plt.subplots()

            img1 = ax.imshow(fdata, vmin=-bounds, vmax=bounds)
            img1.set_cmap(plt.colormaps['bwr'])

            plt.xlim(x - 40, x + 40)
            plt.ylim(y + 60, y - 60)

            plt.ylabel('Time (seconds)')
            plt.xlabel('Distance (Channels)')

            fig.colorbar(img1, label= "Nano Strain per Second [nm/m/s]")

            plt.show(block=False)
            plt.close()


            if input("Noise: ").lower() == "y":
                # cluster file format = c[ymin, ymax, xmin, xmax, points]
                cluster = [y - 30, y + 30, x - 20, x + 20, 0]
                print(f'{cluster}')

                # label file format = l[diffX, diffY, label]
                label = [cluster[1] - cluster[0], cluster[3] - cluster[2], 'n']
                print(f'{label}')

                clusters.append(cluster)
                labels.append(label)
                event_count += 1

        with open(f"{label_directory}/{save}/{time}", "wb") as fp:  # Pickling
            pickle.dump(labels, fp)

        with open(f"{cluster_directory}/{save}/{time}", "wb") as fp:  # Pickling
            pickle.dump(clusters, fp)

        sample_count += 1
        print(f"{sample_count}/27 random noise sampled")

if __name__ == '__main__':
    tdms_directory = "G:/New Data/NovemberNightNew/NightNew"
    cluster_directory = "G:/Clusters/Noise"
    label_directory = "G:/Labeled Clusters and Features/Noise"
    save = 'NNNight'

    noise_sampling(tdms_directory, cluster_directory, label_directory, save)