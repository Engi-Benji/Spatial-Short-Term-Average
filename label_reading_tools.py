import os
import pickle
from clustering import labelled_cluster_association, cluster_association

def associated_labelling(root_folder):

    windows = sorted(os.listdir(f"{root_folder}"))

    unlabelled_windows = 0
    c_count = 0
    p_count = 0
    f_count = 0
    e_count = 0
    n_count = 0

    for window in windows:
        with (open(f"{root_folder}{window}", "rb") as fp):  # Unpickling
            window_import = pickle.load(fp)

            for cluster in window_import:
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

    print(f"Unlabelled windows: {unlabelled_windows}")
    print(f"c count: {c_count}")
    print(f"p count: {p_count}")
    print(f"f count: {f_count}")
    print(f"e count: {e_count}")
    print(f"n count: {n_count}")

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

    # associated_labelling(f"E:/New Data/NovemberNightNew/Assisted Labelled Data/")
    unassociated_labelling_10()
