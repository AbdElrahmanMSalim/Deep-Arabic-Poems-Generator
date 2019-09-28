from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Sequential
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam


# def build_model():
#     model = Sequential()
#     model.add(LSTM(128, input_shape=(40, 42), return_sequences=True))
#     model.add(LSTM(256, return_sequences=True))
#     model.add(LSTM(512, return_sequences=True))
#     model.add(LSTM(1024))
#     model.add(Dense(42, activation='softmax'))

#     optimizer = Adam(lr=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#     model.summary()
#     return model

def build_model(Tx=40, n_x=42):
    model = Sequential()
    model.add(LSTM(128, input_shape=(Tx, n_x), return_sequences=True))
    model.add(LSTM(256))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(LSTM(1024))
    model.add(Dense(n_x, activation='softmax'))

    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()
    return model
