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



class Perceptron(object):
    def __init__(self, batch_size=1, lr=1e-3, num_epoch=100, fit_bias=True):
        super(Perceptron, self).__init__()
        self.fit_bias = fit_bias
        self.w = None
        self.batch_size = batch_size
        self.lr = lr
        self.num_epoch = num_epoch

    def _init_params(self, num_features):
        self.w = np.random.normal(size=(num_features, 1))

    def loader(self, X, Y):
        base = np.c_[X, Y]
        np.random.shuffle(base)
        X, Y = base[:, :-1], base[:, -1]
        data, label = [], []
        for x, y in zip(X, Y):
            data.append(x)
            label.append(y)
            if len(data) == self.batch_size:
                data, label = np.asarray(data), np.asarray(label)
                label = np.expand_dims(label, axis=-1) if len(label.shape) == 1 else label
                yield data, label
                data = []
                label = []

    def _fit_with_sgd(self, X, Y):
        x_y = np.c_[X, Y]
        for epoch in range(self.num_epoch):
            np.random.shuffle(x_y)
            error_num = 0
            for index in range(len(x_y)):
                x, y = x_y[index, :-1], x_y[index, -1]
                if y * self.w.T.dot(x) < 0:
                    dw = -x.dot(y).reshape((-1, 1))
                    self.w -= self.lr * dw

    def fit(self, X, Y):
        if self.fit_bias:
            X = np.c_[X, np.ones_like(Y)]
        self._init_params(X.shape[1])
        self._fit_with_sgd(X, Y)

    def get_params(self):
        if self.fit_bias:
            w = self.w[:-1]
            b = self.w[-1]
        else:
            w = self.w
            b = 0
        return w, b

    def predict(self, X):
        # if self.fit_bias:
        #     X = np.c_[X, np.ones(X.shape[0])]
        v = - X * self.w[0] / self.w[1] - self.w[2] / self.w[1]
        return v

    def draw(self, X):
        plt.scatter(X[:X.shape[0] // 2, 0], X[:X.shape[0] // 2, 1], c="y", s=5)
        plt.scatter(X[X.shape[0] // 2:, 0], X[X.shape[0] // 2:, 1], c="k", s=5)
        plt.plot(X[:X.shape[0] // 2, 0], self.predict(X[:X.shape[0] // 2, 0]))
        plt.show()
