import os
import pickle

if __name__ == '__main__':

    root_folder = f"G:/New Data/NovemberDayNew/Assisted Labelled Data/"
    labels = []

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
                if cluster[2] == "unlabelled" or cluster[2] == 'u':
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
