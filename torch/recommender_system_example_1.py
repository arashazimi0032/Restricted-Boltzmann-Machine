import pandas as pd
import numpy as np
from RBM import RBM
import matplotlib.pyplot as plt

training_set = pd.read_csv(r'..\data\ml-100k\u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv(r'..\data\ml-100k\u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

n_user = np.max([np.max(training_set[:, 0]), np.max(test_set[:, 0])])
n_movies = np.max([np.max(training_set[:, 1]), np.max(test_set[:, 1])])

def convert(data):
    out = []
    for u in range(n_user):
        user_index = data[:, 0] == u + 1
        ind_movie = data[user_index, 1]
        ind_rating = data[user_index, 2]
        rating = np.zeros(n_movies)

        rating[ind_movie - 1] = ind_rating

        out.append(list(rating))
    return np.array(out)

def convert_to_binary_rating(data):
    data[data == 0] = -1
    data[(data > 0) & (data < 3)] = 0
    data[data >= 3] = 1
    return data


training_set = convert_to_binary_rating(convert(training_set))
test_set = convert_to_binary_rating(convert(test_set))

rbm = RBM(nv=n_movies, nh=64, k=20, lr=0.008)

rbm.fit(training_set, batch_size=512, epoch=10, validation_data=test_set, verbose=1)

predicted = rbm.predict(test_set, verbose=1)

pd.DataFrame(rbm.total_losses).plot()
plt.show()
