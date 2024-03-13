import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.5, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_with_zeros(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0

    def propagate(self, X, Y):
        m = X.shape[1]
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        cost = np.mean(-Y * np.log(A) - (1 - Y) * np.log(1 - A))
        dw = np.dot(X, (A - Y).T) / m
        db = np.sum(A - Y) / m
        return {"dw": dw, "db": db}, cost

    def optimize(self, X, Y):
        costs = []
        for i in range(self.num_iterations):
            grads, cost = self.propagate(X, Y)
            dw = grads["dw"]
            db = grads["db"]
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        return costs

    def fit(self, X_train, Y_train):
        self.initialize_with_zeros(X_train.shape[0])
        costs = self.optimize(X_train, Y_train)
        return costs

    def predict(self, X):
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        return np.round(A)

    def accuracy(self, Y_pred, Y_true):
        return 100 - np.mean(np.abs(Y_pred - Y_true)) * 100
