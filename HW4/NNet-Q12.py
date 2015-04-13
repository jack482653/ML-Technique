import random
import math
import numpy as np
from multiprocessing import Pool, Manager
import time
from itertools import izip


def getdata(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            d = line.strip().split(' ')
            data.append(map(float, d))
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
    X = [d[:-1] for d in data_set]
    Y = [d[-1] for d in data_set]
    # c_e = []
    for x, y, u in izip(X, Y, U):
        if algo(p, x) != y:
            # c_e.append(-1)
            error = error + u
        else:
            pass
            # c_e.append(1)
    return error / float(len(Y))


def tanh_d(s):
    return 4.0 * math.exp(2.0 * s) / (math.exp(2.0 * s) + 1) ** 2.0


def NNet(W, x):
    xx = np.matrix([1.0] + x).T
    for w in W:
        xx = w.T * xx
        xx = np.append(np.matrix([1.0]), np.tanh(xx), axis=0)
    return sign(xx[1, 0])


def NNet_train(eta, r, L, train_data):
    T = 50000
    # eta: the learning rate
    # r: range of initializing W
    # L: Level of the Neural Network ex: [2, 3, 1]
    # step 1: initialization of W
    # L with x0: fix L without considering x^(l)_0, 0 <= l <= len(L)
    # tStart = time.time()
    L_w0 = [l + 1 for l in L]
    W = []
    for i, j in izip(L_w0, L[1:]):
        w = [
            [random.uniform(-r, r) for _ in range(j)]
            for _ in range(i)]
        W.append(np.matrix(w))
    # tEnd = time.time()
    # print 'Step1 exec time: {}'.format(tEnd - tStart)
    # step 2: backpropagation
    for i in xrange(T):
        # step 2-1: stocastic-randomly pick [x_n, y_n]
        # tStart = time.time()
        d = random.choice(train_data)
        x_n = np.matrix([1.0] + d[:-1]).T
        y_n = d[-1]
        # tEnd = time.time()
        # print 'Step2-1 exec time: {}'.format(tEnd - tStart)
        # step 2-2: forward-randomly pick [x_n, y_n]
        # tStart = time.time()
        X = [x_n]
        for w in W:
            x = w.T * X[-1]
            x = np.tanh(x)
            X.append(np.append(np.matrix([1.0]), x, axis=0))
        # tEnd = time.time()
        # print 'Step2-2 exec time: {}'.format(tEnd - tStart)
        # step 2-3: backward-compute all delta
        # tStart = time.time()
        Delta = [np.matrix(-2.0 * (y_n - X[-1][1, 0]) * (1.0 - X[-1][1, 0]**2))]
        for l in range(len(L) - 2, 0, -1):
            xx = X[l][1:, 0]
            one = np.matrix(np.ones(xx.shape))
            d = np.multiply(W[l][1:, :] * Delta[0], (one - np.power(xx, 2.0)))
            Delta = [d] + Delta
        # tEnd = time.time()
        # print 'Step2-3 exec time: {}'.format(tEnd - tStart)
        # step 2-4: gradient descent
        # tStart = time.time()
        for l in range(len(L) - 1):
            W[l] = W[l] - eta * X[l] * Delta[l].T
        # tEnd = time.time()
        # print 'Step2-4 exec time: {}'.format(tEnd - tStart)
    return W


def NN500(p, train, test, e_out):
    ID = p[0]
    eta = p[1]
    r = p[2]
    M = p[3]
    # train
    tStart = time.time()
    W = NNet_train(eta, r, M, train)
    tEnd = time.time()
    print '[{}, {}]NNet exec time: {}'.format(r, ID, tEnd - tStart)
    # save result
    tStart = time.time()
    U_test = [1.0 for _ in test]
    e_out[ID] = error_01_u(NNet, W, test, U_test)
    tEnd = time.time()
    print '[{}, {}]Save exec time: {}'.format(r, ID, tEnd - tStart)
    return 0


manager = Manager()

train = manager.list(getdata('hw4_nnet_train.dat'))
test = manager.list(getdata('hw4_nnet_test.dat'))
e_out = {}
experiment = 1
pool = Pool(processes=1)

eta = 0.1
R = [0, 0.001, 0.1, 10.0, 1000.0]
M = [2, 21, 1]
for r in R:
    e_out[r] = manager.list(range(experiment))

print 'Q12'

tStart = time.time()
for r in R:
    for i in range(experiment):
        NN500([i, eta, r, M], train, test, e_out[r])
        pool.apply_async(NN500, [[i, eta, r, M], train, test, e_out[r]])
pool.close()
pool.join()
tEnd = time.time()
print 'Total exec time: {}'.format(tEnd - tStart)

print e_out
for p in e_out:
    print 'parameter: {}, avg_eout: {}'.format(p, avg(e_out[p]))

print 'done'
