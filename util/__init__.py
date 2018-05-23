from scipy.sparse import lil_matrix
import random


def tsv_to_matrix(f, rows=None, cols=None):
    """
    Convert file in tsv format to a csr matrix
    """
    # Read the size of our matrix
    # We know it can't be the best way to do that, but it is the simplest
    user_max_value = rows - 1 if rows is not None else 0
    item_max_value = cols - 1 if cols is not None else 0

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

    return data


def show_matplot_fig():
    """
    Store and show an image from matplotlib that is in the context.

    We are using it to avoid installing several libraries only to show a graphic
    """
    import PIL
    from matplotlib import pyplot as plt

    path = './img/test.png'

    plt.savefig(path)
    img = PIL.Image.open(path)
    img.show()


def split_train_test(file_tsv):
    """
    Split a tsv file into two others:
        1) Train file with 80 perc of the data
        2) Test file with 20 perc of the data
    """
    M = tsv_to_matrix(file_tsv)

    m, _ = M.shape

    train_list = []
    test_list = []

    for i in range(m):
        data = M[i].nonzero()[1]
        random.shuffle(data)

        # 80%
        mark = int(len(data) * 0.8)

        train = data[:mark]
        test = data[mark:]

        train_list.append(train)
        test_list.append(test)

    def store_matrix(data_list, ofile):
        result = []
        for i, data in enumerate(data_list):
            for d in data:
                result.append('%s %s %s' % (i, d, '1.0'))
        result = "\n".join(result)

        f = open(ofile, 'w')
        f.write(result)
        f.close()

    file_without_ext = file_tsv.rsplit('.', 1)[0]
    file_train = '%s_train.tsv' % file_without_ext
    file_test = '%s_test.tsv' % file_without_ext

    store_matrix(train_list, file_train)
    store_matrix(test_list, file_test)
    mark = int(len(data) * 0.8)

    return file_train, file_test


def split_train_validation_test(file_tsv):
    """
    Split a tsv file into two others:
        1) Train file with 80 perc of the data
        2) Test file with 20 perc of the data
    """
    M = tsv_to_matrix(file_tsv)

    m, _ = M.shape

    train_list = []
    validation_list = []
    test_list = []

    for i in range(m):
        data = M[i].nonzero()[1]
        random.shuffle(data)

        # 60%
        mark_train = int(len(data) * 0.6)

        # 20%-20%
        train = data[:mark_train]
        other = data[mark_train:]

        mark_validation = int(len(other) * 0.5)
        validation = other[:mark_validation]
        test = other[mark_validation:]

        train_list.append(train)
        validation_list.append(validation)
        test_list.append(test)

    # TODO: remove this duplication =/
    def store_matrix(data_list, ofile):
        result = []
        for i, data in enumerate(data_list):
            for d in data:
                result.append('%s %s %s' % (i, d, '1.0'))
        result = "\n".join(result)

        f = open(ofile, 'w')
        f.write(result)
        f.close()

    file_without_ext = file_tsv.rsplit('.', 1)[0]
    file_train = '%s_train.tsv' % file_without_ext
    file_validation = '%s_validation.tsv' % file_without_ext
    file_test = '%s_test.tsv' % file_without_ext

    store_matrix(train_list, file_train)
    store_matrix(validation_list, file_validation)
    store_matrix(test_list, file_test)

    return file_train, file_validation, file_test


def cross_split_train_test(file_tsv):
    """
    Split a tsv file into two others (with cross validation):
        1) Train file with 80 perc of the data
        2) Test file with 20 perc of the data
    """
    M = tsv_to_matrix(file_tsv)

    m, _ = M.shape

    fold1, fold2, fold3, fold4, fold5 = [], [], [], [], []

    # Creating folds
    for i in range(m):
        data = M[i].nonzero()[1]
        random.shuffle(data)
        t = len(data)

        f1, f2, f3, f4, f5 = (int(t * .2), int(t * .4), int(t * .6),
                              int(t * .8), t - 1)

        fold1.append(data[:f1])
        fold2.append(data[f1:f2])
        fold3.append(data[f2:f3])
        fold4.append(data[f3:f4])
        fold5.append(data[f4:f5])

    def store_matrix(data_list, ofile):
        result = []
        for i, data in enumerate(data_list):
            for d in data:
                result.append('%s %s %s' % (i, d, '1.0'))
        result = "\n".join(result)

        f = open(ofile, 'w')
        f.write(result)
        f.close()

    folds = [fold1, fold2, fold3, fold4, fold5]

    files_train = []
    file_without_ext = file_tsv.rsplit('.', 1)[0]

    for number in range(len(folds)):
        train_list = [0] * len(folds[0])
        test_list = folds[number]

        for exclude in range(len(folds)):
            if number != exclude:
                for i, l in enumerate(folds[number]):
                    if train_list[i] == 0:
                        train_list[i] = []

                    train_list[i].extend(folds[number][i])

        file_train = '%s_train%s.tsv' % (file_without_ext, number)
        file_test = '%s_test%s.tsv' % (file_without_ext, number)

        store_matrix(train_list, file_train)
        store_matrix(test_list, file_test)

        files_train.append(file_train)
        files_train.append(file_test)

    return file_train, file_test


def mm2csr(M, ofile):
    """
    Convert a matrix to a csr file in the format specified in:
        http://www-users.cs.umn.edu/~xning/slim/html/index.html#examples
    """
    m, n = M.shape

    fhdl = open(ofile, 'w')
    for i in range(m):
        line = []
        for z in M[i].nonzero()[0]:
            # .csr files begins with 1 and not zero, by this we put z+1
            line.append(str(z))
            # TODO: change it, SSLIM only accepts 1 or 0 values
            #line.append(str(M[i][z]))
            line.append("1")
        line = "%s\n" % " ".join(line)
        fhdl.write(line)

    fhdl.close()


def generate_slices(total_columns):
    """
    Generate slices that will be processed based on the number of cores
    available on the machine.
    """
    from multiprocessing import cpu_count

    cores = cpu_count()

    segment_length = total_columns / cores

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


def make_compatible(A, B):
    """
    1) XXX For some reasons generated matrices (side information and base matrix)
    can have a little difference between them (1, 2 or 3 elements). Here we
    shrink the greatest matrix to avoid incompatibilities.

    2) Another important point done
    HACK: for performance, this task is done inplace
    """
    from bisect import bisect_left

    def removecols(W, col_list):
        if min(col_list) == W.shape[1]:
            raise IndexError('column index out of bounds')
        rows = W.rows
        data = W.data

        for i in xrange(W.shape[0]):
            for j in col_list:
                pos = bisect_left(rows[i], j)

                if pos == len(rows[i]):
                    continue
                elif rows[i][pos] == j:
                    rows[i].pop(pos)
                    data[i].pop(pos)
                    if pos == len(rows[i]):
                            continue
                for pos2 in xrange(pos, len(rows[i])):
                    rows[i][pos2] -= 1

        W._shape = (W._shape[0], W._shape[1] - len(col_list))
        return W

    na = A.shape[1]
    nb = B.shape[1]

    if na > nb:
        els_to_remove = []

        for i in reversed(range(na - nb)):
            els_to_remove.append(i + nb)

        removecols(A, els_to_remove)

    elif na < nb:
        els_to_remove = []

        for i in reversed(range(nb - na)):
            els_to_remove.append(i + na)

        removecols(B, els_to_remove)

    return A, B


def normalize_values(M):
    """
    Get all values in matrix and normalize to the minimin being 0 and the
    maximum being 1.
    """
    from sklearn.preprocessing import normalize
    return normalize(M).tolil()
    # --
    value_min = min([min(i) for i in M.data if i])
    value_max = max([max(i) for i in M.data if i])

    if value_min == value_max:
        value_min = 0

    interval = float(value_max - value_min)

    for row in M.data:
        for col in xrange(len(row)):
            row[col] /= interval

    return M


def save_matrix(M, path, binary=False):
    """
    Save a tsv matrix in a desired path.
    """
    width, _ = M.shape

    result = []
    first = True
    f = open(path, 'w')

    for i in xrange(width):
        for j in M[i].nonzero()[-1]:
            if binary:
                result.append('%s %s %s' % (i, j, '1'))
            else:
                result.append('%s %s %s' % (i, j, M[i, j]))

            # We save only non sparse data
            if len(result) > 10000:
                print 'Already Wrote: ', i
                if first:
                    f.write("\n".join(result))
                    first = False
                else:
                    f.write("\n" + "\n".join(result))
                result = []

    # Last save
    if first:
        f.write("\n".join(result))
        first = False
    else:
        f.write("\n" + "\n".join(result))
    result = []

    f.close()


def parse_args(beta=False, side_information=False):
    import argparse
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        help='Matrix file to train the model',
        required=True
    )
    parser.add_argument(
        '--test',
        help='Matrix file to test the model',
        required=True
    )
    if side_information:
        parser.add_argument(
            '--side_information',
            help='Side information to improve learning',
            required=True
        )
    parser.add_argument(
        '--beta',
        type=float,
        help=('Parameter that gives weight to the side_information matrix')
    )
    parser.add_argument(
        '--output',
        help=('File to put the results'),
        required=True
    )

    if side_information:
        parser.add_argument(
            '--normalize',
            type=int,
            help=('Parameter that defines if data '
                  'in side information matrix will be normalized'),
            required=False
        )

    args = parser.parse_args()
    args.beta = args.beta or 0.011
    return args
