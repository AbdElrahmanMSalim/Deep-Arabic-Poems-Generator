from utils import getLettersAndDiacritics, convert_to_two_hot
import sys
import numpy as np
from keras.models import load_model


def generate_output(model, Tx=40, n_x=41):
    generated = ''
    # sentence = text[start_index: start_index + Tx]
    # sentence = '0'*Tx
    # usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
    # zero pad the sentence to Tx characters.
    usr_input = 'ش'

    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    generated += usr_input

    sys.stdout.write("\n\nHere is your poem: \n\n")
    sys.stdout.write(usr_input)
    for i in range(400):
        x_pred = np.zeros((1, Tx, n_x))
        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t] = convert_to_two_hot(char)
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature=1.0)
        next_char = index_to_letter[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

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
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(41), p=probas.ravel())
    return out


letters, diacritics = getLettersAndDiacritics()
# perhabs there is a problem here with the arrangement
lettersAndDiacritics = letters + diacritics
index_to_letter = dict((i, c) for i, c in enumerate(lettersAndDiacritics))

model = load_model(  # such and such)
generate_output(models)
