import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self, batch_size=1, epoch_num=10, lr=1e-2, l1_ratio=None, l2_ratio=None,
                 fit_bias=True, opt="sgd"):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr = lr
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.fit_bias = fit_bias
        self.opt = opt
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



class LDA(object):
    def __init__(self):
        self.w = None

    def fit(self, X1, X2):
        mean_c1 = np.mean(X1, axis=0).reshape((-1, 1))
        mean_c2 = np.mean(X2, axis=0).reshape((-1, 1))
        sub1 = mean_c1 - X1.T
        s_c1 = sub1.dot(sub1.T) / len(X1)
        sub2 = mean_c2 - X2.T
        s_c2 = sub2.dot(sub2.T) / len(X2)
        self.w = np.linalg.pinv(s_c1 + s_c2).dot(mean_c1 - mean_c2)

    def predict(self, data):
        return data.dot(self.w)

    def draw(self, data, label):
        assert len(data.shape) <= 2, "the dimension of data is too large to draw"
        assert len(label.shape) <= 1, "the shape of label should be consistent with axis x"
        new_data = self.predict(data)
        plt.scatter(new_data[:], data[:, 1], c=label, s=5)
        plt.show()


class Logistic_regression(object):
    def __init__(self, batch_size, num_epoch, lr=1e-3, fit_bias=True):
        super(Logistic_regression, self).__init__()
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.lr = lr
        self.fit_bias = fit_bias
        self.w = None

    def init_params(self, num_features):
        self.w = np.random.normal(size=[num_features, 1])

    def loader(self, X, Y):
        data = np.c_[X, Y]
        np.random.shuffle(data)
        n = len(data)
        for i in range(n // self.batch_size + 1):
            it = data[i * self.batch_size:min((i + 1) * self.batch_size, n), :]
            batch_x, batch_y = it[:, :-1], it[:, -1].reshape((-1, 1))
            yield batch_x, batch_y

    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v))

    def _fit_with_gd(self, X, Y):
        for _ in range(self.num_epoch):
            for x, y in self.loader(X, Y):
                try:
                    self.w -= -x.T.dot(y - self.sigmoid(x.dot(self.w))) * self.lr
                    # print(self.w)
                except:
                    print("w:" + str(self.w.shape) + " x:" + str(x.shape) + " y:" + str(y.shape))
                    return

    def fit(self, X, Y):
        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, axis=-1)
        if self.fit_bias:
            X = np.c_[X, np.ones_like(Y)]

        self.init_params(X.shape[-1])
        self._fit_with_gd(X, Y)

    def get_params(self):
        w = self.w
        return w

    def predict(self, X, Y):
        if self.fit_bias:
            X = np.c_[X, np.ones_like(Y)]
        y_prob = self.sigmoid(X.dot(self.w))
        y_hat = np.asarray([1 if p >= 0.5 else 0 for p in y_prob])
        acc = np.sum(y_hat == Y) / len(Y)
        return acc        


class GDA(object):
    def __init__(self):
        self.phi = None
        self.mu_1 = None
        self.mu_2 = None
        self.sigma = None

    def fit(self, x, y):
        # x = (x - np.mean(x, axis=0)) / np.var(x, axis=0)
        # print(x.shape)
        self.phi = np.mean(y)
        n = len(y)
        n1 = n * self.phi
        n2 = n - n1

        self.mu_1 = (x.T.dot(y) / n1).reshape((-1, 1))
        self.mu_2 = (x.T.dot((1 - y)) / n2).reshape((-1, 1))
        s1 = np.zeros((x.shape[-1], x.shape[-1]))
        s2 = np.zeros((x.shape[-1], x.shape[-1]))

        for i in range(n):
            if y[i] == 1:
                s1 += (x[i] - self.mu_1).dot((x[i] - self.mu_1).T)
            else:
                s2 += (x[i] - self.mu_2).dot((x[i] - self.mu_2).T)
        s1 /= n1
        s2 /= n2
        self.sigma = (n1 * s1 + n2 * s2) / n

    def get_params(self):
        return self.phi, self.mu_1, self.mu_2, self.sigma

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def evaluate(self, x, cla):
        prob_1 = []
        prob_2 = []
        for x_i in x:
            x_i = x_i.reshape((-1, 1))
            p1 = 1 / (np.linalg.det(self.sigma) ** 0.5) \
                 * np.exp(-0.5 * (x_i - self.mu_1).T.dot(np.linalg.inv(self.sigma)).dot((x_i - self.mu_1))) * self.phi
            p2 = 1 / (np.linalg.det(self.sigma) ** 0.5) \
                 * np.exp(-0.5 * (x_i - self.mu_2).T.dot(np.linalg.inv(self.sigma)).dot((x_i - self.mu_2))) * (
                             1 - self.phi)
            p1 = self.sigmoid(p1)
            p2 = self.sigmoid(p2)
            prob_1.append(p1)
            prob_2.append(p2)
        label = np.cast["int32"](prob_1 >= prob_2)
        acc = np.sum(label == cla)
        return acc


class NaiveBayesClassifier(object):
    def __init__(self):
        self.positive_sigmas = []
        self.positive_mus = []
        self.negative_sigmas = []
        self.negative_mus = []
        self.phi = None

    def fit(self, X, Y):
        for j in range(X.shape[-1]):
            index = (Y == 1)
            sigma = np.var(X[index, j])
            mu = np.mean(X[index, j])
            self.positive_sigmas.append(sigma)
            self.positive_mus.append(mu)

        for j in range(X.shape[-1]):
            index = (Y == 0)
            sigma = np.var(X[index, j])
            mu = np.mean(X[index, j])
            self.negative_sigmas.append(sigma)
            self.negative_mus.append(mu)

        self.phi = np.mean(Y)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x, y):
        posi_pred = np.ones(x.shape[0])
        nega_pred = np.ones(x.shape[0])

        for j in range(x.shape[1]):
            prob = 1 / np.sqrt(self.positive_sigmas[j]) * np.exp(
                -0.5 / self.positive_sigmas[j] * (x[:, j] - self.positive_mus[j])**2)
            posi_pred *= prob
        for j in range(x.shape[1]):
            prob = 1 / np.sqrt(self.negative_sigmas[j]) * np.exp(
                -0.5 / self.negative_sigmas[j] * (x[:, j] - self.negative_mus[j])**2)
            nega_pred *= prob

        posi_pred = self.sigmoid(posi_pred * self.phi)
        nega_pred = self.sigmoid(nega_pred * (1 - self.phi))
        y_hat = np.cast["int32"](posi_pred >= nega_pred)
        acc = np.sum(y_hat == y) / x.shape[0]
        return acc

    def get_params(self):
        return self.positive_sigmas, self.positive_mus, self.negative_sigmas, self.negative_mus    