"""
cSLIM Parallel implementation. To understand deeply how it works we encourage you to
read "Sparse Linear Methods with Side Information for Top-N Recommendations"
"""
from sklearn.linear_model import SGDRegressor
import numpy as np
from recommender import slim_recommender
from util import tsv_to_matrix, split_train_test
from metrics import compute_precision
import multiprocessing
import ctypes
import sys
from scipy.sparse import vstack

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

        W[j] = w


def generate_slices(total_columns):
    """
    Generate slices that will be processed based on the number of cores
    available on the machine.
    """
    cores = multiprocessing.cpu_count()

    segment_length = total_columns/cores

    ranges = []
    now = 0

    while now < total_columns:
        end = now + segment_length

        # The last part can be a little greater that others in some cases, but
        # we can't generate more than #cores ranges
        end = end if end + segment_length <= total_columns else total_columns
        ranges.append((now, end))
        now = end

    return ranges

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
    B = B[:, :-1]
    Balpha = B

    import pdb;pdb.set_trace()
    #Mline = np.concatenate((A, Balpha))
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


W = sslim_train(A, B)

recommendations = slim_recommender(A, W)

compute_precision(recommendations, test_file)

"""
main('data/atracoes/10/usuarios_atracoes_train.tsv',
     'data/atracoes/10/palavras_atracoes.tsv',
     'data/atracoes/10/usuarios_atracoes_test.tsv')
"""
