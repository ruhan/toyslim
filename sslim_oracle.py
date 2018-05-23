"""
It is an implementation that uses SSLIM to identify all possible items to
recommend to each user, but in the evaluation, we test the results as we had an
"oracle" that know how to order recommendations of SSLIM in to achieve the best
precision value that is possible.

It is very useful to use as an upper boundary of any type of re-ranking
algorithm.
"""
from sslim import sslim_train
from util import tsv_to_matrix, make_compatible, save_matrix
from util.metrics import compute_precision_as_an_oracle
from util.recommender import slim_recommender
from util import parse_args, normalize_values
import simplejson as json


def main(train_file, user_sideinformation_file, test_file, normalize):
    A = tsv_to_matrix(train_file)
    B = tsv_to_matrix(user_sideinformation_file)

    if normalize:
        B = normalize_values(B)

    A, B = make_compatible(A, B)

    W = sslim_train(A, B)

    save_matrix(W, 'sslim_oracle_wmatrix.tsv')
    recommendations = slim_recommender(A, W)

    precisions = compute_precision_as_an_oracle(recommendations, test_file)

    return precisions

args = parse_args(side_information=True)
precisions = main(args.train, args.side_information, args.test, args.normalize)
open(args.output, 'w').write(json.dumps(precisions))
