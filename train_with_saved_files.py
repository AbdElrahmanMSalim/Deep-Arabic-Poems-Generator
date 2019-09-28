import numpy as np
import pandas as pd
import sys
import datetime
from datagenerator import DataGenerator
from build_model import build_model, load_weights
from keras.callbacks import Callback

Tx = 40


def make_dicts():
    partition = {'train': [], 'val': []}
    labels = {'train': [], 'val': []}

    with open('X_train.txt') as f:
        for line in f:
            partition["train"].append(line[:-1])

    with open('X_val.txt') as f:
        for line in f:
            partition["val"].append(line[:-1])

    with open('Y_train.txt') as f:
        for line in f:
            labels["train"].append(line[:-1])

    with open('Y_val.txt') as f:
        for line in f:
            labels["val"].append(line[:-1])

    return partition, labels


class MyCustomCallback(Callback):

    def on_train_batch_begin(self, batch, logs=None):
        print('Training: batch {} begins at {}'.format(
            batch, datetime.datetime.now().time()))

    def on_train_batch_end(self, batch, logs=None):
        print('Training: batch {} ends at {}'.format(
            batch, datetime.datetime.now().time()))

    def on_test_batch_begin(self, batch, logs=None):
        print('Evaluating: batch {} begins at {}'.format(
            batch, datetime.datetime.now().time()))

    def on_test_batch_end(self, batch, logs=None):
        print('Evaluating: batch {} ends at {}'.format(
            batch, datetime.datetime.now().time()))


if __name__ == '__main__':
    model = build_model()

    print('Loading Weights...')
    weights_path = 'weights/my_model_weights2.h5'
    model = load_weights(model, weights_path)

    partition, labels = make_dicts()

    training_generator = DataGenerator(partition['train'], labels['train'])
    validation_generator = DataGenerator(partition['val'], labels['val'])

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=3,
                        epochs=1,
                        verbose=2,
                        callbacks=[MyCustomCallback()])
