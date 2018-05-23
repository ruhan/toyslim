"""
cSLIM basic implementation. To understand deeply how it works we encourage you to
read "Sparse Linear Methods with Side Information for Top-N Recommendations"
"""
from sklearn.linear_model import SGDRegressor
from util.recommender import slim_recommender, normalize_values
from util import tsv_to_matrix, make_compatible, parse_args
from util.metrics import compute_precision
from scipy.sparse import vstack
from scipy.sparse import lil_matrix
import simplejson as json


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
    alpha = l1_reg + l2_reg
    l1_ratio = l1_reg / alpha

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

    Mline = vstack((A, Balpha), format='lil')
    m, n = A.shape

    # Fit each column of W separately
    W = lil_matrix((n, n))

    columns = Mline.shape[1]

    for j in range(columns):
        if j % 50 == 0:
            print '-> %2.2f%%' % ((j / float(columns)) * 100)

        mlinej = Mline[:, j].copy()

        # We need to remove the column j before training
        Mline[:, j] = 0

        model.fit(Mline, mlinej.toarray().ravel())

        # We need to reinstate the matrix
        Mline[:, j] = mlinej

        w = model.coef_

        # Removing negative values because it makes no sense in our approach
        w[w < 0] = 0

        for el in w.nonzero()[0]:
            W[(el, j)] = w[el]

    return W


def main(train_file, user_sideinformation_file, test_file, normalize):
    A = tsv_to_matrix(train_file)
    B = tsv_to_matrix(user_sideinformation_file)

    if normalize:
        B = normalize_values(B)

    A, B = make_compatible(A, B)

    W = sslim_train(A, B)

    recommendations = slim_recommender(A, W)

    return compute_precision(recommendations, test_file)


args = parse_args(side_information=True)
precisions = main(args.train, args.side_information, args.test, args.normalize)
open(args.output, 'w').write(json.dumps(precisions))
