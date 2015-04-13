import numpy as np
from itertools import izip


def getdata(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            d = line.strip().split(' ')
            x = np.array(map(float, d[:-1]))
            y = float(d[-1])
            data.append((x, y))
    return data


def avg(lst):
    return sum(lst) / float(len(lst))


def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0


def error_01_u(algo, p, data_set, U):
    error = 0.0
    X = [d[0] for d in data_set]
    Y = [d[1] for d in data_set]
    # c_e = []
    for x, y, u in izip(X, Y, U):
        if algo(p, x) != y:
            # c_e.append(-1)
            error = error + u
        else:
            pass
            # c_e.append(1)
    return error / float(len(Y))


train = getdata('hw4_knn_train.dat')
test = getdata('hw4_knn_test.dat')
U_train = [1.0 for _ in train]
U_test = [1.0 for _ in test]

def one_NN(p, x):
    global train
    min_distance = 1000000
    min_ty = 0
    if type(x) == list:
        x = np.array(x)
    for tx, ty in train:
        distance = np.dot(x - tx, x - tx)
        if min_distance > distance:
            min_distance = distance
            min_ty = ty
    return min_ty

def kNN(k, x):
    global train
    if type(x) == list:
        x = np.array(x)
    min_distance = [(tx, ty, np.dot(x - tx, x - tx)) for tx, ty in train]
    proto = sorted(min_distance, key = lambda x : x[2])[:k]
    result = 0.0
    for px, py, dis in proto:
        result = result + py * dis
    return sign(result)


print 'Experiment with 1 Nearest Neighbor'
print 'avg e_in: {}'.format(error_01_u(one_NN, _, train, U_train))
print 'avg e_out: {}'.format(error_01_u(one_NN, _, test, U_test))

print 'Experiment with k Nearest Neighbor'
k = 5
print 'avg e_in: {}'.format(error_01_u(kNN, k, train, U_train))
print 'avg e_out: {}'.format(error_01_u(kNN, k, test, U_test))