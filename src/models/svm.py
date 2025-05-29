from scipy.optimize import minimize
import numpy as np

class SVM:
    def __init__(self, data, label, eps=1e-5):
        self.data = data
        self.label = label
        self.eps = eps

    def fun(self):
        n, m = self.data.shape
        if n != len(self.label):
            raise ValueError("Error: data and label should have the same dimension!")
        y = np.repeat(self.label.reshape(n, 1), m, axis=1)
        A = np.multiply(self.data, y)
        A = np.dot(A, A.T)
        return lambda alpha: 0.5 * np.dot(alpha, np.dot(A, alpha)) - np.sum(alpha)

    def createfun(self, i):
        return lambda alpha: alpha[i]

    def con(self):
        C = 10.0
        cons = []
        y = self.label
        for i in range(len(y)):
            cons.append({'type': 'ineq', 'fun': self.createfun(i)})
            cons.append({'type': 'ineq', 'fun': lambda alpha, i=i: C - alpha[i]})
        cons.append({'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)})
        return tuple(cons)

    def fit(self, x0):
        cons = self.con()
        res = minimize(self.fun(), x0, method='SLSQP', constraints=cons)
        if not res.success:
            raise RuntimeError("Optimization failed: " + res.message)

        self.alpha = res.x
        ay = np.multiply(self.alpha, self.label).reshape(-1, 1)
        ay_data = np.multiply(self.data, np.repeat(ay, self.data.shape[1], axis=1))
        self.w = np.sum(ay_data, axis=0)

        sv = self.alpha > self.eps
        self.sv = np.argwhere(sv).flatten()
        y_sv = self.label[sv]
        x_sv = self.data[sv, :]
        b = [1 / y_sv[i] - np.dot(self.w, x_sv[i, :]) for i in range(len(y_sv))]
        self.b = np.mean(b)

    def predict(self, X):
        # X shape: (n_samples, n_features)
        # self.w shape: (n_features,)
        v = np.dot(X, self.w) + self.b
        return np.sign(v)