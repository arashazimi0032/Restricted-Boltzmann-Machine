import numpy as np
import pandas as pd
import torch
from time import time


class RBM:
    def __init__(self, nv, nh, k=10, lr=0.001):
        '''

        :param nv: (int) the size of visible units
        :param nh: (int) the size of hidden units
        :param k: (int, optional) The number of Gibbs sampling. Defaults to 10.
        :param lr: (float, optional) learning rate. Defaults to 0.001.
        '''

        self.nv = nv
        self.nh = nh
        self.k = k
        self.lr = lr

        self.no_data_value = -1
        self.total_losses = {'train': [], 'validation': []}
        self.loss_validation = 0

        self.W = torch.randn(self.nh, self.nv)
        self.a = torch.randn(1, self.nh)
        self.b = torch.randn(1, self.nv)

    def visible_to_hidden_sampling(self, x):
        """
        Conditional sampling a hidden variable given a visible variable.

        :param x: (Tensor): The visible variables.
        :return:
                (Tensor): The hidden probability.
                (Tensor): The hidden variables.
        """

        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def hidden_to_visible_sampling(self, y):
        """
        Conditional sampling a visible variable given a hidden variable.

        :param y: (Tensor): The hidden variables.
        :return:
                (Tensor): The visible probability.
                (Tensor): The visible variables.
        """
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += self.lr * (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += self.lr * torch.sum((v0 - vk), 0)
        self.a += self.lr * torch.sum((ph0 - phk), 0)

    def fit(self, x=None, batch_size=128, epoch=100, validation_data=None, verbose=0, no_data_value=-1):
        x = self.modify_data_type(x)
        self.no_data_value = no_data_value

        for e in range(epoch):
            s_t = time()
            loss = 0
            it = 0
            for row in range(0, len(x) - batch_size, batch_size):
                vk = x[row: row + batch_size]
                v0 = x[row: row + batch_size]
                ph0, _ = self.visible_to_hidden_sampling(v0)
                for k in range(self.k):
                    vk = self.gibbs_sampling(vk)
                    vk[v0 == self.no_data_value] = v0[v0 == self.no_data_value]  # in recommender systems, the -1 meaning is that they are empty.

                phk, _ = self.visible_to_hidden_sampling(vk)
                self.train(v0, vk, ph0, phk)

                loss += self.batch_loss(v0, vk)
                it += 1

            self.total_losses['train'].append(np.array(loss / it).tolist())

            Time = self.strftime(time() - s_t)

            if type(validation_data) != type(None):
                if verbose != 0:
                    print(f'epoch: {e + 1} ==> loss train: {loss / it} ==> time train: {Time} ==> ', end='')
                _ = self.predict(validation_data, verbose=verbose)
                self.total_losses['validation'].append(self.loss_validation)
            else:
                if verbose != 0:
                    print(f'epoch: {e + 1} ==> loss train: {loss / it} ==> time train: {Time}')

        if type(validation_data) == type(None):
            self.total_losses.pop('validation')

        if verbose != 0:
            print('------------- process finished -------------')

    def predict(self, x, verbose=0):
        x = self.modify_data_type(x)
        s_t = time()
        v = x
        vt = x
        if len(vt[vt != self.no_data_value]) > 0:
            v = self.gibbs_sampling(v)
            loss = self.batch_loss(vt, v)
            predicted = v.tolist()
        else:
            predicted = vt.tolist()
            loss = None

        Time = self.strftime(time() - s_t)
        if verbose != 0:
            print(f'loss validation: {loss} ==> time validation: {Time}')
        self.loss_validation = np.array(loss).tolist()
        return np.array(predicted)

    def gibbs_sampling(self, vk):
        _, hk = self.visible_to_hidden_sampling(vk)
        _, vk = self.hidden_to_visible_sampling(hk)
        return vk

    def strftime(self, t):
        h = int(t // 3600)
        m = int((t - h * 3600) // 60)
        s = np.round(t - h * 3600 - m * 60, 2)
        return f'{h}h:{m}min:{s}sec'


    def batch_loss(self, v0, vk):
        return torch.mean(torch.abs(v0[v0 != self.no_data_value] - vk[v0 != self.no_data_value]))

    def modify_data_type(self, x):
        if type(x) == list or type(x) == np.ndarray:
            x = torch.FloatTensor(x)
        elif type(x) == pd.DataFrame or type(x) == pd.Series:
            x = torch.FloatTensor(x.values)
        elif type(x) == torch.Tensor:
            x = x
        else:
            raise Exception('Error: type of data must be list or numpy array or pandas DataFrame or Series!')
        return x
