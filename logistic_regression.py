import numpy as np
from sklearn.metrics import accuracy_score

class Logistic:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        pass
    def _iteration_step(self):
        # put your training code here
        pass

    def train(self, x_train, y_train, tol=1e-4, max_iter=100):
        """Receive the input training data, then learn the model.
        Inputs:
        x_train: np.array, shape (num_samples, num_features)
        y_train: np.array, shape (num_samples, )
        tol: double, optional, the stopping criteria for the weights
        max_iter: int, optional, the maximal number of iteration
        Outputs：
        None
        """
        self.w = np.zeros(x_train.shape[1])
        for _ in range(max_iter):
            last_w = self.w
            self._iteration_step()
            if np.linalg.norm(self.w - last_w) < tol:
                break
        return
    def fit(self, x_train, y_train):
        # alias for train
        self.train(x_train, y_train)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict(self, x_test):
        """Do prediction via the learned model.
        Inputs:
        x_test: np.array, shape (num_samples, num_features)
        Outputs:
        pred: np.array, shape (num_samples, ), probability for positive labels
        """
        pred = np.zeros(x_test.shape[0])
        # put your predicting code here

        return pred
    def predict_proba(self, x_data):
        return self.predict(x_data)