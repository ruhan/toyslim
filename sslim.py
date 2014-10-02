from sklearn.linear_model import SGDRegressor
import numpy as np
from recommender import slim_recommender
from util import tsv_to_matrix, split_train_test
from metrics import compute_precision


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

    # Fit each column of W separately
    W = []

    # Following cSLIM proposal on creating an M' matrix = [ M, FT]
    # * alpha is used to control relative importance of the side information
    #Balpha = np.sqrt(alpha) * B
    Balpha = B

    Mline = np.concatenate((A, Balpha))

    for j in range(Mline.shape[1]):
        mlinej = Mline[:, j].copy()

        # We need to remove the column j before training
        Mline[:, j] = 0

        model.fit(Mline, mlinej.ravel())

        # We need to reinstate the matrix
        Mline[:, j] = mlinej

        w = model.coef_

        # Removing negative values because it makes no sense in our approach
        w[w<0] = 0

        W.append(w)

    return W

def main(train_file, user_sideinformation_file, test_file):
    A = tsv_to_matrix(train_file)
    B = tsv_to_matrix(user_sideinformation_file)

    useritem_featureitem = np.concatenate((A, B))
    import pdb;pdb.set_trace()

    W = sslim_train(A, B)

    recommendations = slim_recommender(A, W)

    compute_precision(recommendations, test_file)


if __name__ == '__main__':
    #main('data/train_100.tsv', 'data/item_side_information_100.tsv', 'data/test_100.tsv')
    #train_file, test_file = split_train_test('data/100_without_stemming/usuarios_atracoes.tsv')
    #main(train_file, 'data/100_without_stemming/palavras_atracoes.tsv', test_file)
    main('data/100_without_stemming/usuarios_atracoes_train.tsv',
         'data/100_without_stemming/palavras_atracoes.tsv',
         'data/100_without_stemming/usuarios_atracoes_test.tsv')
