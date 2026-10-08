import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, losses
from keras.models import Model, Sequential
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split

from CNN_training import data_loading
from copy import deepcopy


class SimpleAutoencoder(Model):
    def __init__(self, latent_dimensions):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = Sequential([
            layers.Input(shape=(1, 121, 81)), #this is going to need editing
            layers.Flatten(),
            layers.Dense(latent_dimensions, activation='relu'),
        ])

        self.decoder = Sequential([
            layers.Dense(121 * 81, activation='tanh'),
            layers.Reshape((1, 121, 81))
        ])

    def call(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded


#make train and test here should be able to reuse existing CNN code
if __name__ == '__main__':

    print("------------ Data Preparation ------------")
    print('     1. Loading Data')

    num_epochs = 50
    width = 80
    height = 120
    use_raw = True

    root_folder = f"G:/CNN Formatted Data - {width} x {height}"
    model_save = f"./models/{width}x{height}_{'raw' if use_raw else 'filtered'}_CNN.pth"

    height += 1
    width += 1

    win_raw, win_filtered, labels = data_loading(root_folder, width, height)

    # ------------- normalisation ---------------
    print('     2. Normalizing Data')

    win_raw_normalised = []
    win_filtered_normalised = []
    for w, f in zip(win_raw, win_filtered):

        minVal = w.min()
        maxVal = w.max()

        w_normalised = (((w - minVal) / (maxVal - minVal)) - 0.5) * 2
        win_raw_normalised.append(w_normalised)

        minVal = f.min()
        maxVal = f.max()

        f_normalised = (((f - minVal) / (maxVal - minVal)) - 0.5) * 2
        win_filtered_normalised.append(f_normalised)

    win_raw_normalised = np.array(win_raw_normalised)
    win_filtered_normalised = np.array(win_filtered_normalised)

    if use_raw:
        win_modified = win_raw_normalised
    else:
        win_modified = win_filtered_normalised


    print('     2. Reshaping Data')
    win_modified = np.array(win_modified).astype(np.float32)
    reshaped_modified = win_modified.reshape(-1, 1, height, width)

    labels = np.array(labels)

    for l in labels:
        if l != "f":
            labels[labels == l] = "p"

    labels = np.array(labels)
    unique = np.unique(labels)
    print(unique)

    for i, u in enumerate(unique):
        labels[labels == u] = i

    labels_enc = to_categorical(labels, num_classes=2)

    # balancing

    print('     3. Balancing Data')
    win_modified_balanced = []
    labels_balanced = []

    countOther = 0
    for w, l in zip(reshaped_modified, labels_enc):
        if l[0] != 1:
            labels_balanced.append(l)
            win_modified_balanced.append(w)
            countOther += 1

    countFoot = 0
    for w, l in zip(reshaped_modified, labels_enc):
        if l[0] == 1:
            if countFoot != countOther:
                labels_balanced.append(l)
                win_modified_balanced.append(w)
                countFoot += 1

    win_modified_balanced = np.array(win_modified_balanced)
    labels_balanced = np.array(labels_balanced)

    # train test split
    # ================
    print('     5. Splitting Data')
    # random seed
    random_seed = 2

    # train validation split
    x_train, x_test, y_train, y_test = train_test_split(win_modified_balanced, labels_balanced,
                                                                             test_size=0.3)


    # encode the data
    latent_dimensions = 64
    autoencoder = SimpleAutoencoder(latent_dimensions)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # visualise the original and decoded images
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    n = 6
    plt.figure(figsize=(12, 6))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(121, 81), cmap='bwr')
        plt.title("Original")
        plt.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(121, 81), cmap='bwr')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.show()