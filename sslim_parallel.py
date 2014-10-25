from sklearn.linear_model import SGDRegressor
import numpy as np
from recommender import slim_recommender
from util import tsv_to_matrix, split_train_test
from metrics import compute_precision
import multiprocessing
import ctypes
import sys

train_file, user_sideinformation_file, test_file = sys.argv[1:]

# Loading matrices
A = tsv_to_matrix(train_file)
B = tsv_to_matrix(user_sideinformation_file)

# Loading shared array to be used in results
shared_array_base = multiprocessing.Array(ctypes.c_double, A.shape[1]**2)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(A.shape[1], A.shape[1])


# We create a work function to fit each one of the columns of our W matrix,
# because in SLIM each column is independent we can use make this work in
# parallel
def work(params, W=shared_array):
    j = params[0]
    M = params[1]
    model = params[2]

    mlinej = M[:, j].copy()

    # We need to remove the column j before training
    M[:, j] = 0

    model.fit(M, mlinej.ravel())

    # We need to reinstate the matrix
    M[:, j] = mlinej

    w = model.coef_

    # Removing negative values because it makes no sense in our approach
    w[w<0] = 0

    W[j] = w

def sslim_train(A, B, l1_reg=0.001, l2_reg=0.0001):
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

    # Following cSLIM proposal on creating an M' matrix = [ M, FT]
    # * alpha is used to control relative importance of the side information
    #Balpha = np.sqrt(alpha) * B
    B = B[:, :-3]
    Balpha = B

    Mline = np.concatenate((A, Balpha))

    # Fit each column of W separately. We put something in each positions of W
    # to allow us direct indexing of each position in parallel
    #W = range(Mline.shape[1])
    separated_tasks = []

    for j in range(Mline.shape[1]):
        separated_tasks.append([j, Mline, model])

    pool = multiprocessing.Pool()
    pool.map(work, separated_tasks)
    pool.close()
    pool.join()

    return shared_array


W = sslim_train(A, B)

recommendations = slim_recommender(A, W)

compute_precision(recommendations, test_file)

"""
main('data/atracoes/10/usuarios_atracoes_train.tsv',
     'data/atracoes/10/palavras_atracoes.tsv',
     'data/atracoes/10/usuarios_atracoes_test.tsv')
"""
