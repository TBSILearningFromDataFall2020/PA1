import numpy as np
from sklearn.metrics import accuracy_score

class Logistic:

    def __init__(self, tol=1e-4, max_iter=100):
        """
        Inputs:
        tol: double, optional, the stopping criteria for the weights
        max_iter: int, optional, the maximal number of iteration
        """
        self.tol = tol
        self.max_iter = max_iter
        pass
    def get_params(self, deep=False):
        return {'tol': self.tol, 'max_iter': self.max_iter}

    def _iteration_step(self, x_train, y_train):
        # put your training code here
        mu = 1 / (1 + np.exp(-x_train @ self.w))
        R = np.diag(mu * (1 - mu))
        self.w += np.linalg.lstsq(x_train.T @ R @ x_train, x_train.T @ (y_train - mu))[0]
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
        for _ in range(self.max_iter):
            last_w = self.w.copy()
            self._iteration_step(x_train, y_train)
            if np.linalg.norm(self.w - last_w) < self.tol:
                break
        return

    def fit(self, x_train, y_train):
        # alias for train
        self.train(x_train, y_train)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict(self, x_test):
        """Predict class labels for samples in x_test
        Inputs:
        x_test: np.array, shape (num_samples, num_features)
        Outputs:
        pred: np.array, shape (num_samples, )
        """
        return np.argmax(self.predict_proba(x_test), axis=1)

    def log_loss(self, x_train, y_train):
        """Negative of Likelihood
        """
        y_expand = np.vstack([1 - y_train, y_train])
        predict_prob = self.predict_proba(x_train)
        predict_prob += self.tol * (predict_prob == 0).astype(np.float)
        return -np.sum(y_expand.T * np.log(predict_prob))

    def predict_proba(self, x_data):
        """Predict class labels for samples in x_test
        Inputs:
        x_data: np.array, shape (num_samples, num_features)
        Outputs:
        pred: np.array, shape (num_samples, n_classes),
              the probability of the sample for each class in the model
        """
        pred = np.zeros([x_data.shape[0], 2])
        # put your predicting code here      
        pred[:, 1] = 1 / (1 + np.exp(- x_data @ self.w))
        pred[:, 0] = 1 - pred[:, 1]
        return pred