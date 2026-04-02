import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.03, iterations=1000, l2_lambda=0):
        self.lr = learning_rate
        self.iterations = iterations
        self.l2_lambda = l2_lambda
        self.w = None
        self.b = None
        self.cost_history = []
        self.test_cost_history = []
        self.X_mean = None
        self.X_std = None

    def feature_scaling(self, X):
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        return (X - self.X_mean) / self.X_std

    def compute_cost(self, X, y, w, b):
        m = X.shape[0]

        cost = 0
        f = np.dot(X, w) + b
        cost = np.sum((f - y) ** 2) / (2 * m)

        reg_cost = 0
        reg_cost = (self.l2_lambda / (2 * m)) * np.sum(w**2)

        total_cost = cost + reg_cost

        return total_cost

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iterations):
            f = np.dot(X, self.w) + self.b

            dj_dw = np.dot(X.T, (f - y)) / m

            dj_db = np.sum(f - y) / m

            self.w = self.w * (1 - self.lr * (self.l2_lambda / m)) - self.lr * dj_dw
            self.b = self.b - self.lr * dj_db

            cost = self.compute_cost(X, y, self.w, self.b)
            self.cost_history.append(cost)

    def predict(self, X):
        return np.dot(X, self.w) + self.b
