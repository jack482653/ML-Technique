import random
import math
import numpy as np
import sys
from threading import Thread, Lock
from Queue import Queue
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


def const(x):
    if type(x) == np.matrix:
        t = np.array(x).flatten()
        if len(t) == 1:
            return t[0]
    return None


def tanh_d(s):
    return 4.0 * math.exp(2.0 * s) / (math.exp(2.0 * s) + 1) ** 2.0


def NNet(W, x):
    xx = np.matrix([1.0] + x)
    for w in W:
        xx = [const(ww * xx.T) for ww in w.T]
        xx = np.matrix([1.0] + map(math.tanh, xx))
    return sign(xx[0, 1])


def NNet_train(eta, r, L, train_data):
    T = 50000
    # eta: the learning rate
    # r: range of initializing W
    # L: Level of the Neural Network ex: [2, 3, 1]
    # step 1: initialization of W
    # L with x0: fix L without considering x^(l)_0, 0 <= l <= len(L)
    L_w0 = [l + 1 for l in L]
    W = []
    for i, j in izip(L_w0, L[1:]):
        w = [
            [random.uniform(-r, r) for _ in range(j)]
            for _ in range(i)]
        W.append(np.matrix(w))
    # step 2: backpropagation
    for i in xrange(T):
        # step 2-1: stocastic-randomly pick [x_n, y_n]
        d = random.choice(train_data)
        x_n = np.matrix([1.0] + d[:-1]).T
        y_n = d[-1]
        # step 2-2: forward-randomly pick [x_n, y_n]
        X = [x_n]
        for w in W:
            x = w.T * X[-1]
            x = np.tanh(x)
            X.append(np.append(np.matrix([1.0]), x, axis=0))
        # step 2-3: backward-compute all delta
        Delta = [np.matrix(-2.0 * (y_n - X[-1][1, 0]) * (1.0 - X[-1][1, 0]**2))]
        for l in range(len(L) - 2, 0, -1):
            xx = X[l][1:, 0]
            one = np.matrix(np.ones(xx.shape))
            d = np.multiply(W[l][1:, :] * Delta[0], (one - np.power(xx, 2.0)))
            Delta = [d] + Delta
        # step 2-4: gradient descent
        for l in range(len(L) - 1):
            W[l] = W[l] - eta * X[l] * Delta[l].T
    return W


class NNThread(Thread):
    def __init__(self, train, test, lock):
        Thread.__init__(self)
        self.train = train
        self.test = test
        self.U_test = [1.0 for _ in self.test]
        self.lock = lock

    def run(self):
        while (True):
            id = threadPool.get()
            if id is not None:
                eta = id[0]
                r = id[1]
                M = id[2]
                e_out = id[3]
                W = NNet_train(eta, r, M, self.train)
                self.lock.acquire()
                e_out.append(
                                    error_01_u(
                                                NNet,
                                                W,
                                                self.test,
                                                self.U_test
                                              )
                                  )
                self.lock.release()
                threadPool.task_done()


train = getdata('hw4_nnet_train.dat')
test = getdata('hw4_nnet_test.dat')

experiment = 1
U_train = [1.0 for _ in train]
U_test = [1.0 for _ in test]

THREAD_NUM = 20
threadPool = Queue(0)
lock = Lock()

threads = [
          NNThread(train, test, lock)
          for _ in range(THREAD_NUM)
          ]
for t in threads:
        t.setDaemon(True)
        t.start()

print 'Q11'
eta = 0.1
r = 0.1
M = [3]
e_out = {3: []}
t1 = time.time()
for m in M:
    for i in range(experiment):
        threadPool.put([eta, r, [2, m, 1], e_out[m]])
threadPool.join()
t2 = time.time()
print 'Exec time: {}'.format(t2 - t1)
for m in M:
    print 'm:{}, eout-avg: {}'.format(m, avg(e_out[m]))

print 'done'
# for m in M:
#     print 'M: {}'.format(m)
#     sys.stdout.flush()
#     e_out = []
#     for _ in range(experiment):
#         W = NNet_train(eta, r, [2, m, 1], train)
#         e_out.append(error_01_u(NNet, W, test, U_test))
#     print 'eout: {}'.format(avg(e_out))
#     sys.stdout.flush()
