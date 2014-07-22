from scipy.sparse import lil_matrix
from sklearn.linear_model import SGDRegressor, ElasticNet
import numpy as np

#M = 11
#N = 761

M = 101
N = 2000

def tsv_to_csr_matrix(f):
    """
    Convert file in tsv format to a csr matrix
    """

    data = lil_matrix((M, N))

    with open(f) as input_file:
        for line in input_file:
            x, y, v = line.split(' ')
            x, y = int(x), int(y)
            v = float(v.strip())
            data[x, y] = v

    return data.tocsr()

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
    n, m = N, M

    # Fit each column of W separately
    W = []

    for j in range(n):
        aj = A.getcol(j)
        # We need to remove the column j before training
        A[:, j] = 0

        model.fit(A.toarray(), aj.toarray().ravel())
        # We need to reinstate the matrix
        A[:, j] = aj

        w = model.coef_
        # Removing zeroes
        w[w<0] = 0
        W.append(w)

    return W

def slim_recommender(A, W):
    """
    Generate the A_hat recommendations matrix
    """
    A_hat = A * W

    recommendations = {}

    # Organizing A_hat matrix to simplify Top-N recommendation
    for u in range(1, M):
        for i in range(1, N):
            v = A_hat[u][i]
            if v > 0:
                # NOTE: it only recommends items that the user haven't rated
                # yet
                # With this if we remove already rated items
                if i not in A[u].nonzero()[1]:
                    if u not in recommendations:
                        recommendations[u] = [(i, v)]
                    else:
                        recommendations[u].append((i, v))

    # Ordering the most probable recommendations by A_hat
    for u in recommendations.iterkeys():
        recommendations[u].sort(reverse=True, cmp=lambda x, y: cmp(x[1], y[1]))

    # Removing W training weights of our recommendations
    for u in recommendations:
        for i, t in enumerate(recommendations[u]):
            recommendations[u][i] = t[0]

    return recommendations

def compute_precision(recommendations, test_file):
    """
    Computes recommendation precision based on a tsv test file.
    """
    ## Computing precision
    # Organizing data
    user_item = {}
    with open(test_file) as test_file:
        for line in test_file:
            u, i, v = line.strip().split(' ')
            u, i = int(u), int(i)
            # TODO: accept float =/
            v = 1
            if u in user_item:
                user_item[u].add(i)
            else:
                user_item[u] = set([i])

    # Computing
    total_users = float(len(recommendations.keys()))
    for at in range(1, 21):
        mean = 0
        for u in recommendations.keys():
            relevants = user_item[u]
            retrieved = recommendations[u][:at]
            precision = len(relevants & set(retrieved))/float(len(retrieved))
            mean += precision

        print 'Average Precision @%s: %s' % (at, (mean/total_users))

def main(train_file, test_file):
    A = tsv_to_csr_matrix(train_file)

    W = slim_train(A)

    recommendations = slim_recommender(A, W)

    compute_precision(recommendations, test_file)

main('wrmf.csv.train.0', 'wrmf.csv.test.0')
#main('wrmf.csv.train.atracoes.0', 'wrmf.csv.test.0')
