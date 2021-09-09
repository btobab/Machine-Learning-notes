import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA


class PCA(object):
    def __init__(self):
        self.eig_vecs = None

    def fit(self, x):
        n = len(x)
        h = np.eye(n) - np.ones((n, 1)).dot(np.ones((1, n))) / n
        s = x.T.dot(h).dot(x) / n
        eig_vals, eig_vecs = LA.eig(s)
        idxs = np.argsort(-eig_vals)
        eig_vals = eig_vals[idxs]
        self.eig_vecs = eig_vecs[:, idxs]

    def predict(self, x):
        out = x.dot(self.eig_vecs)
        return out

    def draw(self, x):
        x_hat = self.predict(x)
        plt.scatter(x_hat[:, 0], x_hat[:, 1], s=5, c="y")
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.show()

