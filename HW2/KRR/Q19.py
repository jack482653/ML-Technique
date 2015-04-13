from __future__ import division
import numpy as np

def getdata(file_name):
    X = []
    Y = []
    with open(file_name, 'r') as file:
        for line in file:
            d = line.strip().split(' ')
            X.append(map(float, d[:-1]))
            Y.append(float(d[-1]))
    return {'X':X, 'Y': Y}


def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0

def const(x):
    if type(x) == np.matrix:
        t = np.array(x).flatten()
        if len(t) == 1:
            return t[0]
    return None

def error_01(algo, p, data_set):
    error = 0.0
    X = data_set['X']
    Y = data_set['Y']
    c_e = []
    for d in zip(X, Y):
        if algo(p, d[0]) != d[1]:
            c_e.append(-1)
            error = error + 1.0
        else:
            c_e.append(1)
    return error / float(len(Y)), c_e


def rbf_kernel(x1, x2, r):
    x1 = np.matrix(x1)
    x2 = np.matrix(x2)
    diff = x1 - x2
    return const(np.exp(-r * np.dot(diff, diff.T)))

def kernel_ridge_reg_train(kernel, p_kernel, p_reg, data_set):
    X = np.matrix(data_set['X'])
    Y = np.matrix(data_set['Y'])
    X_size = len(X)
    # generate K
    K = []
    for x in X:
        k = []
        for y in X:
            k.append(kernel(x, y, p_kernel))
        K.append(k)
    K = np.matrix(K)
    b = np.linalg.inv(p_reg * np.identity(X_size) + K) * Y.T
    return b

def kernel_ridge_reg(p, x):
    # p = [kernel, p_kernel, x, b]
    kernel = p[0]
    r = p[1]
    X = p[2]
    b = p[3]
    y = 0.0
    for xx, bb in zip(X, b):
        y = y + bb * kernel(np.matrix(xx), np.matrix(x), r)
    return sign(y)

r = [32.0, 2.0, 0.125]
l = [0.001, 1.0, 1000.0]

data = getdata('hw2_lssvm_all.dat')

train = {'X': data['X'][:400], 'Y':data['Y'][:400]}
test = {'X': data['X'][400:], 'Y':data['Y'][400:]}

for rr in r:
    for ll in l:
        b = kernel_ridge_reg_train(rbf_kernel, rr, ll, train)
        ein, _ = error_01(kernel_ridge_reg, [rbf_kernel, rr, train['X'], b], train)
        eout, _ = error_01(kernel_ridge_reg, [rbf_kernel, rr, train['X'], b], test)
        print 'r:{} l:{} ein:{} eout:{}'.format(rr, ll, ein, eout)