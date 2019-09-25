import numpy as np

batch_size = 64

extenstion = '.npy'
default_path = 'data/vectorized/'
names = ['X_train', 'X_val', 'Y_train', 'Y_val']
# names = ['X_val']

for name in names:
    x = np.load(default_path + name + extenstion)
    divided = np.array_split(x, len(x)//batch_size)
    IDs = []

    for i in range(len(x)//batch_size):
        Id = default_path + 'splitted/' + name + \
            '/' + name + "{0:0=5d}".format(i) + extenstion
        IDs.append(Id)
        np.save(Id, divided[i])

    with open('data/batchesNames/' + name + '.txt', 'w') as f:
        for Id in IDs:
            f.write(Id + '\n')
