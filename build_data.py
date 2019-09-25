import numpy as np
import pandas as pd
from utils import convert_to_two_hot

Tx = 40
n_x = 41
batch_size = 64

extention = '.npy'
default_vectorized_path = 'data/vectorized/'
default_actual_path = 'data/actual/'
batches_names_path = 'data/batchesNames/'


def build_data(text, Tx, stride=3):
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
    Y = []

    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])
        Y.append(text[i + Tx])

    print('number of training examples:', len(X))

    return X, Y


def split_data(X, y, split_at=.05):
    m = len(X)
    split_point = int((1-split_at)*m)
    X_train, Y_train = X[:split_point], y[:split_point]
    X_val, Y_val = X[split_point:], y[split_point:]

    print("X_train shape= ", len(X_train))
    print("Y_train shape= ", len(Y_train))
    print("X_val shape= ", len(X_val))
    print("Y_val shape= ", len(Y_val))

    data = {'X_train': X_train, 'Y_train': Y_train,
            "X_val": X_val, "Y_val": Y_val}

    return data


def load_csv(path):
    df = pd.read_csv('data/cleaned_arabic_dataset.csv')
    print("shape of data", df.shape)
    print("length of first bayt", len(df['Bayt_Text'][0]))
    print("head\n", df.head())
    return df


def vectorize(data, labels):
    X = np.empty((len(data), Tx, n_x), dtype=np.bool)
    y = np.empty((len(data), n_x), dtype=np.bool)

    for i, sentence in enumerate(data):
        two_hot_vec = np.array(np.squeeze(
            convert_to_two_hot(sentence)), dtype='bool')
        X[i, :len(two_hot_vec)] = two_hot_vec
        y[i, :] = np.array(np.squeeze(
            convert_to_two_hot(labels[i])), dtype='bool')
    return X, y


def vectorize_all(non_vectorized_data):
    X_train = non_vectorized_data['X_train']
    Y_train = non_vectorized_data['Y_train']
    X_val = non_vectorized_data['X_val']
    Y_val = non_vectorized_data['Y_val']

    X_train_vectorized, Y_train_vectorized = vectorize(X_train, Y_train)
    X_val_vectorized, Y_val_vectorized = vectorize(X_val, Y_val)

    data = {'X_train_vectorized': X_train_vectorized,
            'Y_train_vectorized': Y_train_vectorized,
            "X_val_vectorized": X_val_vectorized,
            "Y_val_vectorized": Y_val_vectorized}

    return data


def save_data(data, dir_path):
    for name, value in data.items():
        path = dir_path + name + extention
        np.save(path, value)


def get_splitted_vectorized_output_path(name):
    return default_vectorized_path + 'splitted/' + name + '/' + name


def get_splitted_data_text_file_path(name):
    return batches_names_path + name + '.txt'


def split_npy_data(data, batch_size, output_path):
    number_of_batches = len(data)//batch_size
    print("Number of batches: ", number_of_batches)
    divided = np.array_split(data, number_of_batches)
    IDs = []

    for i in range(number_of_batches):
        Id = output_path + "{0:0=5d}".format(i) + extention
        IDs.append(Id)
        np.save(Id, divided[i])

    return IDs


def create_splitted_data_text_file(path, IDs):
    with open(path, 'w') as f:
        for Id in IDs:
            f.write(Id + '\n')


def save_numpy_files(vectorized_data, non_vectorized_data, save_as_batches=True):
    if save_as_batches:
        print("Saving batches...")
        del non_vectorized_data
        for name, value in vectorized_data.items():
            print("Splitting " + name + "...")
            splitted_output_path = get_splitted_vectorized_output_path(name)
            IDs = split_npy_data(value, batch_size, splitted_output_path)

            print("Creating IDs file...")
            batches_output_path = get_splitted_data_text_file_path(name)
            create_splitted_data_text_file(batches_output_path, IDs)
    else:
        print("Saving Vectorized...")
        save_data(vectorized_data, default_vectorized_path)
        print("Saving Non Vectorized...")
        save_data(non_vectorized_data, default_actual_path)


def run(split_at=.3, save_as_batches=True):
    print('Loading CSV file...')
    df = load_csv('cleaned_arabic_dataset.csv')

    print('Building data...')
    X, y = build_data("".join(df['Bayt_Text']), Tx, 7)

    print('Splitting data...')
    non_vectorized_data = split_data(X, y, split_at)

    del X, y

    print('Vectorizing data...')
    vectorized_data = vectorize_all(non_vectorized_data)

    print('Saving data...')
    save_numpy_files(vectorized_data, non_vectorized_data, save_as_batches)


run(split_at=.05, save_as_batches=True)
