"""
SLIM basic implementation. To understand deeply how it works we encourage you to
read "SLIM: Sparse LInear Methods for Top-N Recommender Systems".
"""
from sklearn.linear_model import SGDRegressor
from util import tsv_to_matrix
from metrics import compute_precision
from recommender import slim_recommender
import numpy as np


def slim_train(A, l1_reg=0.001, l2_reg=0.0001):
    """
    Computes W matrix of SLIM

    This link is useful to understand the parameters used:

        http://web.stanford.edu/~hastie/glmnet_matlab/intro.html

        Basically, we are using this:

            Sum( yi - B0 - xTB) + ...
        As:
            Sum( aj - 0 - ATwj) + ...

    Remember that we are wanting to learn wj. If you don't undestand this
    mathematical notation, I suggest you to read section III of:

        http://glaros.dtc.umn.edu/gkhome/slim/overview
    """
    alpha = l1_reg+l2_reg
    l1_ratio = l1_reg/alpha

    model = SGDRegressor(
        penalty='elasticnet',
        fit_intercept=False,
        alpha=alpha,
        l1_ratio=l1_ratio
    )

    # TODO: get dimensions in the right way
    m, n = A.shape

    # Fit each column of W separately
    W = []

    for j in range(n):
        aj = A[:, j].copy()
        # We need to remove the column j before training
        A[:, j] = 0

        model.fit(A, aj.ravel())
        # We need to reinstate the matrix
        A[:, j] = aj

        w = model.coef_
        # Removing zeroes
        w[w<0] = 0
        W.append(w)

    return np.array(W)


def main(train_file, test_file):
    A = tsv_to_matrix(train_file)

    W = slim_train(A)

    recommendations = slim_recommender(A, W)

    compute_precision(recommendations, test_file)

if __name__ == '__main__':
    main('data/train_100.tsv', 'data/test_100.tsv')
