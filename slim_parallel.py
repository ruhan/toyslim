"""
SLIM Parallel implementation. To understand deeply how it works we encourage you to
read "SLIM: Sparse LInear Methods for Top-N Recommender Systems".
"""
from sklearn.linear_model import SGDRegressor
from util import tsv_to_matrix, generate_slices
from metrics import compute_precision
from recommender import slim_recommender
import numpy as np
import multiprocessing
import ctypes
import sys

train_file, test_file = sys.argv[1:]

# Loading matrices
A = tsv_to_matrix(train_file)

# Loading shared array to be used in results
shared_array_base = multiprocessing.Array(ctypes.c_double, A.shape[1]**2)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(A.shape[1], A.shape[1])

# We create a work function to fit each one of the columns of our W matrix,
# because in SLIM each column is independent we can use make this work in
# parallel
def work(params, W=shared_array):
    from_j = params[0]
    to_j = params[1]
    M = params[2]
    model = params[3]
    counter = 0

    for j in range(from_j, to_j):
        counter += 1
        if counter % 10 == 0:
            print 'Range %s -> %s: %2.2f%%' % (from_j, to_j,
                            (counter/float(to_j - from_j)) * 100)
        mlinej = M[:, j].copy()

        # We need to remove the column j before training
        M[:, j] = 0

        model.fit(M, mlinej.toarray().ravel())

        # We need to reinstate the matrix
        M[:, j] = mlinej

        w = model.coef_

        # Removing negative values because it makes no sense in our approach
        w[w<0] = 0

        for el in w.nonzero()[0]:
            W[(el, j)] = w[el]



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
        l1_ratio=l1_ratio,
    )

    total_columns = A.shape[1]
    ranges = generate_slices(total_columns)
    separated_tasks = []

    for from_j, to_j in ranges:
        separated_tasks.append([from_j, to_j, A, model])

    pool = multiprocessing.Pool()
    pool.map(work, separated_tasks)
    pool.close()
    pool.join()

    return shared_array

W = slim_train(A)

recommendations = slim_recommender(A, W)

compute_precision(recommendations, test_file)
