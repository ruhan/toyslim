"""
cSLIM Parallel implementation. To understand deeply how it works we encourage you to
read "Sparse Linear Methods with Side Information for Top-N Recommendations"
"""
from sklearn.linear_model import SGDRegressor
import numpy as np
from recommender import slim_recommender
from util import (tsv_to_matrix, split_train_test, generate_slices,
                  make_compatible, normalize_values, save_matrix)
from metrics import compute_precision
import multiprocessing
import ctypes
import sys
from scipy.sparse import vstack
import argparse
import datetime
import json

print '>>> Start: %s' % datetime.datetime.now()

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Matrix file to train the model',
                                        required=True)
parser.add_argument('--test', help='Matrix file to test the model',
                                        required=True)
parser.add_argument('--side_information',
                                        help='Side information to improve learning', required=True)
parser.add_argument('--beta', type=float, help=('Parameter that gives weight to the '
                                              'side_information matrix'))
parser.add_argument('--output', help=('File to put the results'), required=True)
parser.add_argument('--normalize', type=int, help=('Parameter that defines if data'
                    'in side information matrix will be normalized'),
                    required=False)
parser.add_argument('--fold', type=int, help=('Parameter only to mark the fold '
                    'that is being calculated'),
                    required=False)
args = parser.parse_args()

train_file = args.train
user_sideinformation_file = args.side_information
output = args.output
test_file = args.test
beta = args.beta or 0.011
normalize = args.normalize
fold = args.fold or 0

# Loading matrices
A = tsv_to_matrix(train_file)
B = tsv_to_matrix(user_sideinformation_file)

if normalize:
    B = normalize_values(B)

A, B = make_compatible(A, B)

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


def sslim_train(A, B, l1_reg=0.001, l2_reg=0.0001, beta=0.0011):
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

    # Following cSLIM proposal on creating an M' matrix = [ M, FT ]
    # * beta is used to control relative importance of the side information
    Balpha = beta * B
    Mline = vstack((A, Balpha), format='lil')

    # Fit each column of W separately. We put something in each positions of W
    # to allow us direct indexing of each position in parallel
    total_columns = Mline.shape[1]
    ranges = generate_slices(total_columns)
    separated_tasks = []

    for from_j, to_j in ranges:
        separated_tasks.append([from_j, to_j, Mline, model])

    pool = multiprocessing.Pool()
    pool.map(work, separated_tasks)
    pool.close()
    pool.join()

    return shared_array


W = sslim_train(A, B, beta=beta)

save_matrix(W, user_sideinformation_file.replace('.tsv', 'beta%s_fold%s.Wmatrix.tsv' % (beta, fold)))

del B

recommendations = slim_recommender(A, W)

precisions = compute_precision(recommendations, test_file)

print '>>> End: %s' % datetime.datetime.now()

o = open(output, 'w')
o.write(json.dumps(precisions))
o.close()
