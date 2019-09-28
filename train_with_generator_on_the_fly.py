from datagenerator_creation import DataGenerator
import numpy as np
import pandas as pd
import sys
import datetime
from build_model import build_model
from keras.callbacks import Callback


def split_data(df, split_at=.05):
    m = len(df)
    split_point = int((1-split_at)*m)

    train = "".join(df['Bayt_Text'][:split_point])
    val = "".join(df['Bayt_Text'][split_point:])

    return train, val


path = 'data/cleaned_arabic_dataset.csv'
df = pd.read_csv(path)
train, val = split_data(df)

train_generator = DataGenerator(train, batch_size=64, Tx=40, n_x=42, stride=3)
val_generator = DataGenerator(val, batch_size=64, Tx=40, n_x=42, stride=3)

model = build_model(Tx=40, n_x=42)

# print("Loading saved weights...")
# weights_path = 'weights/my_model_weights2.h5'
# model.load_weights(weights_path)

model.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    use_multiprocessing=True,
                    workers=1,
                    epochs=1,
                    verbose=1)
