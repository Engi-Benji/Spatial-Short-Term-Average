import os
import pickle
from clustering import labelled_cluster_association, cluster_association
import numpy as np

def labelled_data_loading(root_folder, width, height):
    #load in the data
    folders = sorted(os.listdir(root_folder), reverse=False)

    labels = []
    win_raw = []
    win_filtered = []

    for folder in folders:

        windows = sorted(os.listdir(f"{root_folder}/{folder}"))

        for window in windows:
            with (open(f"{root_folder}/{folder}/{window}", "rb") as fp):  # Unpickling
                window_import = pickle.load(fp)

                labels.append(window_import[0])
                win_raw.append(np.array(window_import[1]))
                win_filtered.append(np.array(window_import[2]))

    #remove irregular sized file
    temp_raw = []
    temp_filtered = []
    temp_labels = []

    for (w, f, l) in zip(win_raw, win_filtered, labels):
        if np.shape(w)[0] == height and np.shape(w)[1] == width:
            temp_raw.append(w)
            temp_filtered.append(f)
            temp_labels.append(l)

    win_raw = np.array(temp_raw)
    win_filtered = np.array(temp_filtered)
    labels = temp_labels

    print(f"Total Samples: {len(win_raw)}")
    return win_raw, win_filtered, labels

def associated_labelling(root_folder):

    windows = sorted(os.listdir(f"{root_folder}"))

    unlabelled_windows = 0
    c_count = 0
    p_count = 0
    f_count = 0
    e_count = 0
    n_count = 0
    dupecount = 0

    prev_cluster = None

    for window in windows:
        with (open(f"{root_folder}{window}", "rb") as fp):  # Unpickling
            window_import = pickle.load(fp)

            for cluster in window_import:

                if cluster == prev_cluster:
                                print(f"{root_folder}{window}")
                                print(f"{cluster} == {prev_cluster}")
                                dupecount += 1

                if cluster[2] == "unlabelled" or cluster[2] == 'u' or cluster[2] == '':
                    unlabelled_windows += 1
                elif cluster[2] == 'c':
                    c_count += 1
                elif cluster[2] == 'p':
                    p_count += 1
                elif cluster[2] == 'f':
                    f_count += 1
                elif cluster[2] == 'e':
                    e_count += 1
                elif cluster[2] == 'n':
                    n_count += 1
                else:
                    print(f'Unrecognized label: {cluster[2]}')

                prev_cluster = cluster

    print(f"Unlabelled windows: {unlabelled_windows}")
    print(f"c count: {c_count}")
    print(f"p count: {p_count}")
    print(f"f count: {f_count}")
    print(f"e count: {e_count}")
    print(f"n count: {n_count}")
    print(f"dupecount: {dupecount}")

def unassociated_labelling_10():
    width = 80
    height = 120
    root_folder = f"E:/CNN Formatted Data - {width} x {height}/FNight"

    labels = []
    windows = sorted(os.listdir(root_folder))
    for window in windows:
        with (open(f"{root_folder}/{window}", "rb") as fp):  # Unpickling
            window_import = pickle.load(fp)

            labels.append(window_import[0])

    unlabelled_windows = 0
    c_count = 0
    p_count = 0
    f_count = 0
    e_count = 0
    n_count = 0

    for label in labels:
        if label == "unlabelled" or label == 'u' or label == '':
            unlabelled_windows += 1
        elif label == 'c':
            c_count += 1
        elif label == 'p':
            p_count += 1
        elif label == 'f' or label == 'fa':
            f_count += 1
        elif label == 'e':
            e_count += 1
        elif label == 'n':
            n_count += 1
        else:
            print(f'Unrecognized label: {label}')

    print(f"Unlabelled windows: {unlabelled_windows}")
    print(f"c count: {c_count}")
    print(f"p count: {p_count}")
    print(f"f count: {f_count}")
    print(f"e count: {e_count}")
    print(f"n count: {n_count}")
        

if __name__ == '__main__':

    associated_labelling(f"E:/New Data/NovemberDayNew/Assisted Labelled Data/")
    # unassociated_labelling_10()