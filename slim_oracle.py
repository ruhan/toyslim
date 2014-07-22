"""
It is an implementation that uses SLIM to identify all possible items to
recommend to each user, but in the evaluation, we test the results as we had an
"oracle" that know how to order recommendations of SLIM in to achieve the best
precision value that is possible.

It is very useful to use as an upper boundary of any type of re-ranking
algorithm.
"""
from slim import slim_train
from util import tsv_to_matrix
from metrics import compute_precision_as_an_oracle
from recommender import slim_recommender


def main(train_file, test_file):
    A = tsv_to_matrix(train_file)

    W = slim_train(A)

    recommendations = slim_recommender(A, W)

    compute_precision_as_an_oracle(recommendations, test_file)

if __name__ == '__main__':
    main('data/train_100.tsv', 'data/test_100.tsv')
