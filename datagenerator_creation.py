import numpy as np
import pandas as pd
import keras
from utils import convert_to_two_hot


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, df_text, batch_size=64, Tx=40, n_x=42, stride=3):
        'Initialization'
        self.df_text = df_text
        self.Tx = Tx
        self.n_x = n_x
        self.stride = stride
        self.batch_size = batch_size
        self.text_in_batch_size = 3 * (self.batch_size-1) + self.Tx
        self.data = None
        self.labels = None

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.df_text) / self.text_in_batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'

        current_data = self.df_text[idx *
                                    self.text_in_batch_size:(idx + 1) * self.text_in_batch_size]

        X, y = self.build_data(current_data)
        X, y = self.vectorize(X, y)

        return X, y

    def build_data(self, text):
        """
        Create a training set by scanning a window of size Tx over the text corpus, with stride 3.

        Arguments:
        text -- string, the whole corpus
        Tx -- sequence length, number of time-steps (or characters) in one training example
        stride -- how much the window shifts itself while scanning

        Returns:
        X -- list of training examples
        Y -- list of training labels
        """

        X = []
        y = []

        for i in range(0, len(text) - self.Tx, self.stride):
            X.append(text[i: i + self.Tx])
            y.append(text[i + self. Tx])

        return X, y

    def vectorize(self, data, labels):
        X = np.zeros((len(data), self.Tx, self.n_x), dtype=np.bool)
        y = np.zeros((len(data), self.n_x), dtype=np.bool)

        for i, sentence in enumerate(data):
            two_hot_vec = np.array(np.squeeze(
                convert_to_two_hot(sentence)), dtype='bool')
            X[i, :len(two_hot_vec)] = two_hot_vec
            y[i, :] = np.array(np.squeeze(
                convert_to_two_hot(labels[i])), dtype='bool')

        return X, y
