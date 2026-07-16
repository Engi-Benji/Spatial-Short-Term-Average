import pickle

from TDMS_Batch_Reader import *
from TDMS_Utilities import get_data
from clustering import labelled_cluster_association, cluster_association
from filters import filter_waterfall

def cnn_event_windows_10(tdms_folder, clusters_folder, save, window_width, window_height):
    filenames = sorted([filename for filename in os.listdir(tdms_folder)])

    tdms_array = load_folder(tdms_folder)
    tdms_array = sort_array(tdms_array)

    clusters_array = sorted(os.listdir(clusters_folder))

    total_clusters = 0

    for file_number, tdms in enumerate(tdms_array):

        if file_number == 0:

            #tdms_data = getData(tdms)

            with open(f"{clusters_folder}/{clusters_array[file_number]}", "rb") as fp:   # Unpickling
                cluster_data = pickle.load(fp)

            temp = []
            for c in cluster_data:
                if 20 <= c[3] - c[2] < 100 and c[1] != c[0] > 0:
                    temp.append(c)

            cluster_data = temp
            del temp

            cluster_data = cluster_association(cluster_data, 0.25, 40, 1000, -1, 100, -1, 1000)

            total_clusters += len(cluster_data)

            for event_number, cluster in enumerate(cluster_data):
                #time, channel, point count
                label = 'u'
                sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2)
                channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)

                #TDMS chan chan samp samp
                w = get_data(tdms, (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)), (sample_midp + int(window_height/2)))
                fw = filter_waterfall(w, 1000, 100, -1)

                export = [label, w, fw]

                with open(f"{save}/{label}-{event_number}-{filenames[file_number]}", "wb") as fp:  # Pickling
                    pickle.dump(export, fp)
        else:

            #tdms_data = np.append(getData(tdms_array[file_number - 1]), getData(tdms), axis=0)

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

            for event_number, cluster in enumerate(cluster_data):
                #time, channel, point count
                label = 'u'
                sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2) - 20
                channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)


                if (sample_midp - int(window_height/2)) > 9999:
                    #second file
                    w = get_data(tdms_array[file_number], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)) - 10000, (sample_midp + int(window_height/2)) - 10000)

                elif (sample_midp + int(window_height/2)) < 10000:
                    #first file
                    w = get_data(tdms_array[file_number - 1], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)), (sample_midp + int(window_height/2)))

                else:
                    #in the overlap
                    pt1 = get_data(tdms_array[file_number-1], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), first_sample = (sample_midp - int(window_height/2)))

                    pt2 = get_data(tdms_array[file_number], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), last_sample = (sample_midp + int(window_height/2))-10000)

                    w = np.append(pt1, pt2, axis=0)
                    del pt1, pt2

                fw = filter_waterfall(w, 1000, 100, -1)

                export = [label, w, fw]

                with open(f"{save}/{label}-{event_number}-{filenames[file_number]}", "wb") as fp:  # Pickling
                    pickle.dump(export, fp)

def labelled_cnn_event_windows_10(tdms_folder, labels_folder, clusters_folder, save, window_width, window_height):

    tdms_array = load_folder(tdms_folder)
    tdms_array = sort_array(tdms_array)
    filenames = sorted([filename for filename in os.listdir(tdms_folder)])

    label_array = sorted(os.listdir(labels_folder))

    clusters_array = sorted(os.listdir(clusters_folder))

    total_clusters = 0

    for file_number, tdms in enumerate(tdms_array):

        if file_number == 0:

            #tdms_data = getData(tdms)

            with open(f"{clusters_folder}/{clusters_array[file_number]}", "rb") as fp:   # Unpickling
                cluster_data = pickle.load(fp)

            temp = []
            for c in cluster_data:
                if 20 <= c[3] - c[2] < 100 and c[1] != c[0] > 0:
                    temp.append(c)

            cluster_data = temp
            del temp

            with open(f"{labels_folder}/{label_array[file_number]}", "rb") as fp:   # Unpickling
                label_data = pickle.load(fp)

            cluster_data = cluster_association(cluster_data, 0.25, 40, 1000, -1, 100, -1, 1000)
            cluster_data, label_data = labelled_cluster_association(cluster_data, label_data, 0.25, 40, 1000, -1, 100, -1, 1000)

            total_clusters += len(cluster_data)

            for event_number, (cluster, label) in enumerate(zip(cluster_data, label_data)):
                #time, channel, point count

                sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2)
                channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)

                #TDMS chan chan samp samp
                w = get_data(tdms, (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)), (sample_midp + int(window_height/2)))
                fw = filter_waterfall(w, 1000, 100, -1)

                export = [label, w, fw]

                with open(f"{save}/{label}-{event_number}-{filenames[file_number]}", "wb") as fp:  # Pickling
                    pickle.dump(export, fp)
        else:

            #tdms_data = np.append(getData(tdms_array[file_number - 1]), getData(tdms), axis=0)

            with open(f"{labels_folder}/{label_array[file_number]}", "rb") as fp:   # Unpickling
                label_data = pickle.load(fp)

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
            cluster_data, label_data = labelled_cluster_association(cluster_data, label_data, 0.25, 40, 1000, -1, 100, -1, 1000)

            for event_number, (cluster, label) in enumerate(zip(cluster_data, label_data)):
                #time, channel, point count

                sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2) - 20
                channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)


                if (sample_midp - int(window_height/2)) > 9999:
                    #second file
                    w = get_data(tdms_array[file_number], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)) - 10000, (sample_midp + int(window_height/2)) - 10000)

                elif (sample_midp + int(window_height/2)) < 10000:
                    #first file
                    w = get_data(tdms_array[file_number - 1], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)), (sample_midp + int(window_height/2)))

                else:
                    #in the overlap
                    pt1 = get_data(tdms_array[file_number-1], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), first_sample = (sample_midp - int(window_height/2)))

                    pt2 = get_data(tdms_array[file_number], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), last_sample = (sample_midp + int(window_height/2))-10000)

                    w = np.append(pt1, pt2, axis=0)
                    del pt1, pt2

                fw = filter_waterfall(w, 1000, 100, -1)

                export = [label, w, fw]

                with open(f"{save}/{label}-{event_number}-{filenames[file_number]}", "wb") as fp:  # Pickling
                    pickle.dump(export, fp)

def cnn_event_windows_30(tdms_folder, clusters_folder, save, window_width, window_height):

    filenames = sorted([filename for filename in os.listdir(tdms_folder)])
    tdms_array = load_folder(tdms_folder)
    tdms_array = sort_array(tdms_array)

    label_array = sorted(os.listdir(labels_folder))

    clusters_array = sorted(os.listdir(clusters_folder))

    prior_cInfo = []
    total_clusters = 0


    for file_number, tdms in enumerate(tdms_array):

        future_prior = []
        combined = []

        if file_number == 0:

            #tdms_data = getData(tdms)

            for j in range(5):
                with open(f"{clusters_folder}{clusters_array[file_number+j]}", "rb") as fp:  # Unpickling
                    cInfo = pickle.load(fp)

            if j == 0:
                for c in cInfo:
                    combined.append([c[0], c[1], c[2], c[3], c[4]])
            elif j == 4:
                combined.extend(x for x in cInfo if x not in combined)
                for c in cInfo:
                    future_prior.append([c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]])
            else:
                combined.extend(x for x in cInfo if x not in combined)

        cluster_data = combined
        del combined

        temp = []
        for c in cluster_data:
            if 20 <= c[3] - c[2] < 100 and c[1] != c[0] > 0:
                temp.append(c)

        cluster_data = temp
        del temp

        with open(f"{labels_folder}/{label_array[file_number]}", "rb") as fp:   # Unpickling
            label_data = pickle.load(fp)

        cluster_data = cluster_association(cluster_data, 0.25, 40, 1000, -1, 100, -1, 1000)

        total_clusters += len(cluster_data)

        for event_number, (cluster, label) in enumerate(zip(cluster_data, label_data)):
            #time, channel, point count

            sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2)
            channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)

            #TDMS chan chan samp samp
            w = get_data(tdms, (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)), (sample_midp + int(window_height/2)))
            fw = filter_waterfall(w, 1000, 100, -1)

            # fig1 = plt.figure()
            # img1 = plt.imshow(w, cmap='bwr', vmin=-1000, vmax=1000)
            # plt.title(f'{label[2]}')
            # fig1.show()
            #
            # fig1 = plt.figure()
            # img1 = plt.imshow(fw, cmap='bwr', vmin=-1000, vmax=1000)
            # plt.title(f'{label[2]}')
            # fig1.show()

            export = [label[2], w, fw]

            with open(f"{save}/{label[2]}-{event_number}-{filenames[file_number]}", "wb") as fp:  # Pickling
                pickle.dump(export, fp)

        else:

            with open(f"{labels_folder}/{label_array[file_number]}", "rb") as fp:   # Unpickling
                label_data = pickle.load(fp)

            clust_num = ((file_number + 1) * 6) - 1

            combined = []
            for j in range(1, 6):
                with open(f"{clusters_folder}{clusters_array[clust_num - j]}", "rb") as fp:  # Unpickling
                    cInfo = pickle.load(fp)

                if j == 0:
                    for c in cInfo:
                        combined.append([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]])
                if j == 1:
                    for c in cInfo:
                        combined.extend([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] for c in cInfo if [c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] not in combined)
                        future_prior.append([c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]])
                else:
                    combined.extend([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] for c in cInfo if [c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] not in combined)

            with open(f"{clusters_folder}{clusters_array[clust_num - 6]}", "rb") as fp:  # Unpickling
                cInfo = pickle.load(fp)

            combined.extend([c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]] for c in cInfo if [c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]] not in combined)
            combined = (x for x in combined if x not in prior_cInfo)

            cluster_data = combined
            del combined

            temp = []
            for c in cluster_data:
                if 20 <= c[3] - c[2] < 100 and c[1] != c[0] > 0:
                    temp.append(c)

            cluster_data = temp
            del temp

            cluster_data = cluster_association(cluster_data, 0.25, 40, 1000, -1, 100, -1, 1000)

            for event_number, (cluster, label) in enumerate(zip(cluster_data, label_data)):
                #time, channel, point count

                sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2) - 20
                channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)


                if (sample_midp - int(window_height/2)) > 9999:
                    #second file
                    w = get_data(tdms_array[file_number], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)) - 10000, (sample_midp + int(window_height/2)) - 10000)

                elif (sample_midp + int(window_height/2)) < 10000:
                    #first file
                    w = get_data(tdms_array[file_number - 1], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)), (sample_midp + int(window_height/2)))

                else:
                    #in the overlap
                    pt1 = get_data(tdms_array[file_number-1], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), first_sample = (sample_midp - int(window_height/2)))

                    pt2 = get_data(tdms_array[file_number], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), last_sample = (sample_midp + int(window_height/2))-10000)

                    w = np.append(pt1, pt2, axis=0)
                    del pt1, pt2

                fw = filter_waterfall(w, 1000, 100, -1)

                # fig1 = plt.figure()
                # img1 = plt.imshow(w, cmap='bwr', vmin=-1000, vmax=1000)
                # plt.title(f'{label[2]}')
                # fig1.show()
                #
                # fig1 = plt.figure()
                # img1 = plt.imshow(fw, cmap='bwr', vmin=-1000, vmax=1000)
                # plt.title(f'{label[2]}')
                # fig1.show()

                export = [label, w, fw]

                with open(f"{save}/{label}-{event_number}-{filenames[file_number]}", "wb") as fp:  # Pickling
                    pickle.dump(export, fp)

        prior_cInfo = future_prior

def labelled_cnn_event_windows_30(tdms_folder, labels_folder, clusters_folder, save, window_width, window_height):
    tdms_array = load_folder(tdms_folder)
    tdms_array = sort_array(tdms_array)
    filenames = sorted([filename for filename in os.listdir(tdms_folder)])

    label_array = sorted(os.listdir(labels_folder))

    clusters_array = sorted(os.listdir(clusters_folder))


    total_clusters = 0
    new_total_clusters = 0
    new_cluster_stats = []

    priordata = None
    prior_fdata = None
    prior_cInfo = None

    for file_number, tdms in enumerate(tdms_array):

        future_prior = []
        combined = []

        if file_number == 0:

            #tdms_data = getData(tdms)

            for j in range(5):
                with open(f"{clusters_folder}{clusters_array[file_number+j]}", "rb") as fp:  # Unpickling
                    cInfo = pickle.load(fp)

                if j == 0:
                    for c in cInfo:
                        combined.append([c[0], c[1], c[2], c[3], c[4]])
                elif j == 4:
                    combined.extend(x for x in cInfo if x not in combined)
                    for c in cInfo:
                        future_prior.append([c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]])
                else:
                    combined.extend(x for x in cInfo if x not in combined)

            cluster_data = combined
            del combined

            temp = []
            for c in cluster_data:
                if 20 <= c[3] - c[2] < 100 and c[1] != c[0] > 0:
                    temp.append(c)

            cluster_data = temp
            del temp

            with open(f"{labels_folder}/{label_array[file_number]}", "rb") as fp:   # Unpickling
                label_data = pickle.load(fp)

            if window == "JDay" or window == "FNight" or window == "FDay":
                cluster_data = cluster_association(cluster_data, 0.25, 40, 1000, -1, 100, -1, 1000)


            cluster_data, label_data = labelled_cluster_association(cluster_data, label_data, 0.25, 40, 1000, -1, 100, -1, 1000)

            total_clusters += len(cluster_data)

            for event_number, (cluster, label) in enumerate(zip(cluster_data, label_data)):
                #time, channel, point count

                sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2)
                channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)

                #TDMS chan chan samp samp
                w = get_data(tdms, (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)), (sample_midp + int(window_height/2)))
                fw = filter_waterfall(w, 1000, 100, -1)

                # fig1 = plt.figure()
                # img1 = plt.imshow(w, cmap='bwr', vmin=-1000, vmax=1000)
                # plt.title(f'{label[2]}')
                # fig1.show()
                #
                # fig1 = plt.figure()
                # img1 = plt.imshow(fw, cmap='bwr', vmin=-1000, vmax=1000)
                # plt.title(f'{label[2]}')
                # fig1.show()

                export = [label, w, fw]

                with open(f"{save}/{label}-{event_number}-{filenames[file_number]}", "wb") as fp:  # Pickling
                    pickle.dump(export, fp)

        else:

            #tdms_data = np.append(getData(tdms_array[file_number - 1]), getData(tdms), axis=0)

            with open(f"{labels_folder}/{label_array[file_number]}", "rb") as fp:   # Unpickling
                label_data = pickle.load(fp)

            clust_num = ((file_number + 1) * 6) - 1

            combined = []
            for j in range(1, 6):
                with open(f"{clusters_folder}{clusters_array[clust_num - j]}", "rb") as fp:  # Unpickling
                    cInfo = pickle.load(fp)

                if j == 0:
                    for c in cInfo:
                        combined.append([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]])
                if j == 1:
                    for c in cInfo:
                        combined.extend([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] for c in cInfo if [c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] not in combined)
                        future_prior.append([c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]])
                else:
                    combined.extend([c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] for c in cInfo if [c[0] + 10000, c[1] + 10000, c[2], c[3], c[4]] not in combined)

            with open(f"{clusters_folder}{clusters_array[clust_num - 6]}", "rb") as fp:  # Unpickling
                cInfo = pickle.load(fp)

            combined.extend([c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]] for c in cInfo if [c[0] - 20000, c[1] - 20000, c[2], c[3], c[4]] not in combined)
            combined = (x for x in combined if x not in prior_cInfo)

            cluster_data = combined
            del combined

            temp = []
            for c in cluster_data:
                if 20 <= c[3] - c[2] < 100 and c[1] != c[0] > 0:
                    temp.append(c)

            cluster_data = temp
            del temp

            if window == "JDay" or window == "FNight" or window == "FDay":
                cluster_data = cluster_association(cluster_data, 0.25, 40, 1000, -1, 100, -1, 1000)

            cluster_data, label_data = labelled_cluster_association(cluster_data, label_data, 0.25, 40, 1000, -1, 100, -1, 1000)

            for event_number, (cluster, label) in enumerate(zip(cluster_data, label_data)):
                #time, channel, point count

                sample_midp = cluster[0] + int((cluster[1] - cluster[0])/2) - 20
                channel_midp = cluster[2] + int((cluster[3] - cluster[2])/2)


                if (sample_midp - int(window_height/2)) > 9999:
                    #second file
                    w = get_data(tdms_array[file_number], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)) - 10000, (sample_midp + int(window_height/2)) - 10000)

                elif (sample_midp + int(window_height/2)) < 10000:
                    #first file
                    w = get_data(tdms_array[file_number - 1], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), (sample_midp - int(window_height/2)), (sample_midp + int(window_height/2)))

                else:
                    #in the overlap
                    pt1 = get_data(tdms_array[file_number-1], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), first_sample = (sample_midp - int(window_height/2)))

                    pt2 = get_data(tdms_array[file_number], (channel_midp - int(window_width/2)), (channel_midp + int(window_width/2)), last_sample = (sample_midp + int(window_height/2))-10000)

                    w = np.append(pt1, pt2, axis=0)
                    del pt1, pt2

                fw = filter_waterfall(w, 1000, 100, -1)

                # fig1 = plt.figure()
                # img1 = plt.imshow(w, cmap='bwr', vmin=-1000, vmax=1000)
                # plt.title(f'{label[2]}')
                # fig1.show()
                #
                # fig1 = plt.figure()
                # img1 = plt.imshow(fw, cmap='bwr', vmin=-1000, vmax=1000)
                # plt.title(f'{label[2]}')
                # fig1.show()

                export = [label, w, fw]

                with open(f"{save}/{label}-{event_number}-{filenames[file_number]}", "wb") as fp:  # Pickling
                    pickle.dump(export, fp)

        prior_cInfo = future_prior

if __name__ == '__main__':

    device = "G"
    window = "FNight"
    save = F"{device}:\\CNN Formatted Data - 80 x 120\\{window}"

    tdms_folder = f"{device}:/1000Hz Data/{window}/"
    labels_folder = f"{device}:/Labeled Clusters and Features/{window}/"
    clusters_folder = f"{device}:/Clusters/{window}/"

    if window == "FDay" or window == "FNight":
        labelled_cnn_event_windows_30(tdms_folder, labels_folder, clusters_folder, save, 80, 120)
    else:
        labelled_cnn_event_windows_10(tdms_folder, labels_folder, clusters_folder, save, 80, 120)


    # tdms_directory = "G:/1000Hz Data/NDay/"
    # cluster_directory = "G:/New Data/NovemberSanity/Clusters/"
    # save = "G:/New Data/NovemberSanity/CNN Data"
    #
    # cnn_event_windows_10(tdms_directory, cluster_directory, save, 80, 120)