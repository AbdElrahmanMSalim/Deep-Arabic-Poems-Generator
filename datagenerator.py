import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, labels, shuffle=True):
        'Initialization'
        self.labels = labels
        self.data = data
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.data)

    def __getitem__(self, index):
        'Generate one batch of data'
        X = np.load(self.data[index])
        y = np.load(self.labels[index])

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
