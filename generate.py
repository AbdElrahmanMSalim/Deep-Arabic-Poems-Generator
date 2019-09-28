import sys
import numpy as np
from keras.models import load_model
from utils import getLettersAndDiacritics, convert_to_two_hot, convert_indexes_to_char


def generate_output(model, Tx=40, n_x=42):
    generated = ''
    usr_input = 'بلب'

    sentence = ('{0:0<' + str(Tx) + '}').format(usr_input).lower()
    generated += usr_input

    sys.stdout.write("\n\nHere is your poem: \n\n")
    sys.stdout.write(usr_input)
    for i in range(100):
        x_pred = np.zeros((1, Tx, n_x))
        two_hot_vec = np.squeeze(convert_to_two_hot(sentence.split('0')[0]))
        x_pred[0][:len(two_hot_vec)] = two_hot_vec

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature=1.0)
        next_char = convert_indexes_to_char(next_index)

        generated += next_char
        sentence = sentence + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
        if next_char == '\n':
            continue


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(100, preds, 1)/100
    out1 = np.random.choice(range(42), 1, p=probas.ravel())
    out2 = np.random.choice(range(42), 1, p=probas.ravel())
    dia = [0, 38, 39, 40, 41, 42]
    while(out1 == out2 or (out1 in dia and out2 in dia)):
        out2 = np.random.choice(range(42), 1, p=probas.ravel())

    return [out1[0], out2[0]]


letters, diacritics = getLettersAndDiacritics()
# perhabs there is a problem here with the arrangement
lettersAndDiacritics = letters + diacritics
index_to_letter = dict((i, c) for i, c in enumerate(lettersAndDiacritics))

model = load_model('my_model')
generate_output(model)
