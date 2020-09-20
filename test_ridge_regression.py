import unittest
import numpy as np

from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.ridge import Ridge
from ridge_regression import RidgeRegression

class TestRidgeModel(unittest.TestCase):
    def test_ridge(self):
        # Ridge regression convergence test, compare to the true value
        rng = np.random.RandomState(0)
        alpha = 1.0

        # With more samples than features
        n_samples, n_features = 6, 5
        y = rng.randn(n_samples)
        X = rng.randn(n_samples, n_features)

        ridge = Ridge(alpha=alpha, fit_intercept=False)
        custom_implemented_ridge = RidgeRegression(alpha=alpha)
        ridge.fit(X, y)
        custom_implemented_ridge.fit(X, y)
        self.assertEqual(custom_implemented_ridge.w.shape, (X.shape[1], ))
        self.assertAlmostEqual(ridge.score(X, y), custom_implemented_ridge.score(X, y))

    def test_ridge_singular(self):
        # test on a singular matrix
        rng = np.random.RandomState(0)
        n_samples, n_features = 6, 6
        y = rng.randn(n_samples // 2)
        y = np.concatenate((y, y))
        X = rng.randn(n_samples // 2, n_features)
        X = np.concatenate((X, X), axis=0)

        ridge = RidgeRegression(alpha=0)
        ridge.train(X, y)
        self.assertGreater(ridge.score(X, y), 0.9)

    def test_ridge_vs_lstsq(self):
        # On alpha=0., Ridge and ordinary linear regression yield the same solution.
        rng = np.random.RandomState(0)
        # we need more samples than features
        n_samples, n_features = 5, 4
        y = rng.randn(n_samples)
        X = rng.randn(n_samples, n_features)

        ridge = RidgeRegression(alpha=0)
        ols = LinearRegression(fit_intercept=False)

        ridge.fit(X, y)
        ols.fit(X, y)
        self.assertEqual(ridge.w.shape, ols.coef_.shape)
        for i in range(ols.coef_.shape[0]):
            self.assertAlmostEqual(ridge.w[i], ols.coef_[i])

if __name__ == '__main__':
    unittest.main()