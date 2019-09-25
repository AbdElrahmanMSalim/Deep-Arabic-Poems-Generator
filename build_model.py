from keras.layers import Dense, Activation, Dropout, Input, Masking, Embedding
from keras.callbacks import LambdaCallback
from keras.models import load_model, Sequential
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils import Sequence


def build_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(40, 41), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(1024))
    model.add(Dense(41, activation='softmax'))

    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()
    return model


def load_weights(model, weights_path):
    print("loading saved weights")
    model.load_weights(weights_path)
    return model
