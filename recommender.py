from scipy.sparse import csr_matrix

def slim_recommender(A, W):
    """
    Generate the A_hat recommendations matrix
    """
    # XXX: I don't know why, but we need to use csr conversion here. Without it
    # we are receiving a ValueError
    A_hat = csr_matrix(A) * W

    recommendations = {}
    m, n = A.shape

    # Organizing A_hat matrix to simplify Top-N recommendation
    for u in range(1, m):
        for i in range(1, n):
            v = A_hat[(u,i)]
            if v > 0:
                # NOTE: it only recommends items that the user haven't rated yet
                # Because that, we ignore already rated items
                if A[(u, i)] == 0:
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


