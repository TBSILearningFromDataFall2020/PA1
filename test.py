import unittest
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal

from ridge_regression import RidgeRegression
from logistic_regression import Logistic

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
        assert_array_almost_equal(ridge.w, ols.coef_)

class TestLogisticModel(unittest.TestCase):
    def test_binary(self):
        # Test logistic regression on a binary problem.
        iris = load_iris()
        target = (iris.target > 0).astype(np.intp)
        # target = np.array(["setosa", "not-setosa"])[target]

        clf = Logistic(max_iter=2000)
        clf.fit(iris.data, target)

        self.assertEqual(clf.w.shape, (iris.data.shape[1],))
        self.assertTrue(clf.score(iris.data, target) > 0.9)

    def test_logistic_iris(self):
        # Test logistic regression with the iris dataset
        iris = load_iris()

        n_samples, n_features = iris.data.shape

        target = iris.target_names[iris.target]

        # Test that OvR solvers handle
        # multiclass data correctly and give good accuracy
        # score (>0.95) for the training data.
        clf = OneVsRestClassifier(Logistic())
        clf.fit(iris.data, target)
        assert_array_equal(np.unique(target), clf.classes_)

        pred = clf.predict(iris.data)
        self.assertTrue(np.mean(pred == target) > .95)

        probabilities = clf.predict_proba(iris.data)
        assert_array_almost_equal(probabilities.sum(axis=1),
                                np.ones(n_samples))

        pred = iris.target_names[probabilities.argmax(axis=1)]
        self.assertTrue(np.mean(pred == target) > .95)

    def test_convergence_rate(self):
        # shows that IRLS converges faster than one-order method
        X, y = make_classification(n_samples=20, random_state=0)
        lr1 = LogisticRegression(random_state=0, max_iter=3,
                             solver='liblinear')
        lr1.fit(X, y)
        clf = Logistic(max_iter=3)
        clf.fit(X, y)
        self.assertTrue(lr1.score(X, y) < clf.score(X, y))
        pass

if __name__ == '__main__':
    unittest.main()