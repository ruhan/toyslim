from cdecimal import Decimal
PRECISION_AT = 20

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
    total_users = Decimal(len(recommendations.keys()))
    for at in range(1, PRECISION_AT+1):
        mean = 0
        for u in recommendations.keys():
            relevants = user_item[u]
            retrieved = recommendations[u][:at]
            precision = len(relevants & set(retrieved))/Decimal(len(retrieved))
            mean += precision

        print 'Average Precision @%s: %s' % (at, (mean/total_users))


def compute_precision_as_an_oracle(recommendations, test_file):
    """
    Computes recommendation precision based on a tsv test file but computing it
    as we had an oracle that predicts the best ranking that can be done with
    the recommendations.
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

    total_users = Decimal(len(recommendations.keys()))

    # Changing recommendations as an ORACLE
    for u in recommendations.keys():
        recommendations[u] = list(user_item[u] & set(recommendations[u]))
        recommendations[u] += list(
            user_item[u] | set(recommendations[u]) -
            (user_item[u] & set(recommendations[u]))
        )

    # Computing
    for at in range(1, PRECISION_AT+1):
        mean = 0
        for u in recommendations.keys():
            relevants = user_item[u]
            retrieved = recommendations[u][:at]
            precision = len(relevants & set(retrieved))/Decimal(len(retrieved))
            mean += precision

        print 'Average Precision @%s: %s' % (at, (mean/total_users))

