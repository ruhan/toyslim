from scipy.sparse import lil_matrix

def tsv_to_matrix(f, rows=None, cols=None):
    """
    Convert file in tsv format to a csr matrix
    """
    # Read the size of our matrix
    # We know it can't be the best way to do that, but it is the simplest
    user_max_value = rows-1 if rows is not None else 0
    item_max_value = cols-1 if cols is not None else 0

    with open(f) as input_file:
        for line in input_file:
            u, i, _ = line.split(' ')
            u, i = int(u), int(i)

            if u > user_max_value:
                user_max_value = u

            if i > item_max_value:
                item_max_value = i

    # Building the matrix
    data = lil_matrix((user_max_value + 1, item_max_value + 1))

    with open(f) as input_file:
        for line in input_file:
            x, y, v = line.split(' ')
            x, y = int(x), int(y)
            v = float(v.strip())
            data[x, y] = v

    return data.toarray()


