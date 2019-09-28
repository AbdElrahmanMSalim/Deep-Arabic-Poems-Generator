
# coding: utf-8

# # Two Hot Vector

# In[1]:


import numpy as np
from pyarabic.araby import strip_tashkeel, separate


# In[2]:


def map_dicts(letters):
    letter_to_index = dict((c, i) for i, c in enumerate(letters))
    index_to_letter = dict((i, c) for i, c in enumerate(letters))

    return letter_to_index, index_to_letter


# In[3]:


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


# In[4]:


def convert_letter_to_index(word, is_letter):
    indices = []
    for i, ch in enumerate(word):
        if(is_letter == True):
            indices.append(letter_to_index[ch])
        else:
            indices.append(diacritic_to_index[ch])
    return np.array(indices)


# In[5]:


def convert_word_to_one_hot(word, is_letter):
    if (is_letter == True):
        letters_count = 37
    else:
        letters_count = 5

    indices = convert_letter_to_index(word, is_letter)
    return convert_to_one_hot(np.array(indices), letters_count)


# In[6]:


def convert_to_two_hot(word):
    if(len(word) == 1 and word[0] in diacritics):
        letters_hot_vector = np.zeros((1, 37))
        diacritics_hot_vector = convert_word_to_one_hot(word, is_letter=False)
    else:
        without_diacritics, only_diacritics = separate(word)
        only_diacritics = only_diacritics.replace("ـ", "ْ")

        letters_hot_vector = convert_word_to_one_hot(
            without_diacritics, is_letter=True)
        diacritics_hot_vector = convert_word_to_one_hot(
            only_diacritics, is_letter=False)
#     print(diacritics_hot_vector.shape)
#     print(letters_hot_vector.shape)
    return np.concatenate([letters_hot_vector, diacritics_hot_vector], axis=1)


def convert_to_char(two_hot_vec):
    indexes = np.where(two_hot_vec == 1)[0]
    return index_to_letter[indexes[0]] + index_to_letter[indexes[1]]

# In[7]:


def getLettersAndDiacritics():
    letters = [' ', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث',
               'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
               'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي']
    diacritics = ['َ', 'ُ', 'ِ', 'ْ', '']
    return letters, diacritics


letters, diacritics = getLettersAndDiacritics()


# In[8]:


def get_unique_letters(df):
    l = df['Bayt_Text']
    letters = set()
    for bayt in l:
        for j in bayt:
            letters.add(j)
    return letters


# In[72]:
def convert_vec_to_char(two_hot_vec):
    indexes = np.where(two_hot_vec == 1)[0]
    return index_to_letter[indexes[0]] + index_to_letter[indexes[1]]

# In[73]:


def convert_indexes_to_char(indexes):
    return index_to_letter[indexes[0]] + index_to_letter[indexes[1]]


# In[74]:
letter_to_index, index_to_letter = map_dicts(letters)
diacritic_to_index, index_to_diacritic = map_dicts(diacritics)
