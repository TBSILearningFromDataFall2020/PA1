import numpy as np
from sklearn.metrics import r2_score

class RidgeRegression:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        pass

    def train(self, x_train, y_train):
        """Receive the input training data, then learn the model.
        Inputs:
        x_train: np.array, shape (num_samples, num_features)
        y_train: np.array, shape (num_samples, )
        Outputs：
        None
        """
        self.w = np.zeros(x_train.shape[1])
        # put your training code here
        n = x_train.shape[1]
        A = x_train.T @ x_train + self.alpha * np.identity(n)
        b = x_train.T @ y_train
        self.w = np.linalg.lstsq(A, b, rcond=-1)[0]
        return
    def fit(self, x_train, y_train):
        # alias for train
        self.train(x_train, y_train)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def predict(self, x_test):
        """Do prediction via the learned model.
        Inputs:
        x_test: np.array, shape (num_samples, num_features)
        Outputs:
        pred: np.array, shape (num_samples, )
        """

        pred = x_test.dot(self.w)

        return pred