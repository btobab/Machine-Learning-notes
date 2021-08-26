import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self, batch_size=1, epoch_num=10, lr=1e-2, l1_ratio=None, l2_ratio=None,
                 fit_bias=True, opt="sgd", if_standard=False):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr = lr
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.fit_bias = fit_bias
        self.opt = opt
        self.if_standard = if_standard
        self.w = None

    def _fit_closed(self, x, y):
        if self.l1_ratio is None and self.l2_ratio is None:
            self.w = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
        elif self.l1_ratio is None and self.l2_ratio is not None:
            self.w = np.linalg.pinv(x.T.dot(x) + self.l2_ratio * np.eye(x.shape[1])).dot(x.T).dot(y)
        else:
            self._fit_with_sgd(x, y)

    def loader(self, x, y):
        for _ in range(x.shape[0] // self.batch_size):
            idxs = np.random.choice(np.arange(x.shape[0]), size=self.batch_size)
            batch_x = x[idxs]
            batch_y = y[idxs]
            yield batch_x, batch_y

    def init_params(self, feature_num):
        self.w = np.random.normal(size=(feature_num, 1), scale=3)

    def _fit_with_sgd(self, x, y):
        for _ in range(self.epoch_num):
            for data, label in self.loader(x, y):
                dw = 2 * data.T.dot(data.dot(self.w) - label)
                penalty = np.zeros(shape=(x.shape[1] - 1, 1))
                if self.l1_ratio is not None:
                    penalty += self.l1_ratio * self.w[:-1]
                if self.l2_ratio is not None:
                    penalty += 2 * self.l2_ratio * self.w[:-1]
                penalty = np.concatenate([penalty, np.asarray([[0]])], axis=0)
                dw += penalty

                self.w -= self.lr * dw / self.batch_size

    def fit(self, x, y):
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=-1)
        if self.if_standard:
            self.feature_mean = np.mean(x, axis=0)
            self.feature_std = np.std(x, axis=0)
            x = (x - self.feature_mean) / self.feature_std
            self.label_mean = np.mean(y, axis=0)
            self.label_std = np.std(y, axis=0)
            y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)
        if self.fit_bias:
            x = np.c_[x, np.ones_like(y)]
        self.init_params(x.shape[1])
        if self.opt is None:
            self._fit_closed(x, y)
        elif self.opt == "sgd":
            self._fit_with_sgd(x, y)

    def get_params(self):
        if self.fit_bias:
            w = self.w[:-1].tolist()
            b = self.w[-1].tolist()
        else:
            w = self.w.tolist()
            b = 0

        # if self.if_standard:
        #     w = w * self.label_std / self.feature_std

        return w, b

    def predict(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        if self.fit_bias:
            x = np.c_[x, np.ones(shape=(x.shape[0], 1))]
        y = x.dot(self.w)
        return y

    def draw(self, x, y, dim=0):
        plt.scatter(x[:, dim], y, s=3, c="y")
        plt.plot(x[:, dim], self.predict(x))
        plt.show()
