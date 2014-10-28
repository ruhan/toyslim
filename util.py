from scipy.sparse import lil_matrix
import random


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
        data = M[i].nonzero()[0]
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
