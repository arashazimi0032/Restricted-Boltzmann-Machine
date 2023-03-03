from mlxtend.data import loadlocal_mnist
from RBM import RBM
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X, _ = loadlocal_mnist(images_path='../data/minist/t10k-images.idx3-ubyte',
                       labels_path='../data/minist/t10k-labels.idx1-ubyte')
X = X.astype(float) / 255
X[X > 0] = 1

X_train, X_test = train_test_split(X, test_size=0.3, random_state=101)

rbm = RBM(nv=len(X[0]), nh=128, k=5, lr=0.01)

rbm.fit(X_train, batch_size=128, epoch=150, verbose=1, validation_data=X_test)

pred = rbm.predict(X_test, verbose=1)

pd.DataFrame(rbm.total_losses).plot()

plt.figure(figsize=(12, 12))
for i in range(1, 33):
    plt.subplot(8, 8, i * 2 - 1)
    plt.imshow(pred[i].reshape(28, 28))
    plt.subplot(8, 8, i * 2)
    plt.imshow(X_test[i].reshape(28, 28))

plt.figure(figsize=(12, 12))
for i in range(1, 65):
    plt.subplot(8, 8, i)
    plt.imshow(rbm.W[i-1].numpy().reshape(28, 28))

plt.show()

