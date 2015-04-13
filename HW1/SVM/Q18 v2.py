import numpy as np
import itertools as it
import math
from sklearn import svm
import pandas as P

kernel = 'rbf'
gamma = 100.0
C = [0.001, 0.01, 0.1, 1, 10]


def readfile(filename):
    X = []
    Y = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.decode('utf-8').strip().split(u' ')
            x = map(float, line)
            X = X + [x[1:]]
            Y = Y + [1.0 if x[0] == 0.0 else -1.0]
    return np.array(X), np.array(Y)


def cal_cosi(clf, X, Y):
    margin = Y * clf.decision_function(X).T
    return [1.0 - m if m < 1.0 else 0.0 for m in margin[0]]


def obj(clf):
    def kernel(X, Y, r):
        return math.exp(-r * np.dot(X - Y, (X - Y).T))
    SV = clf.support_vectors_
    ay = clf.dual_coef_[0]
    result = sum(
        [p1[0] * p2[0] * kernel(p1[1], p2[1], gamma)
            for p1, p2 in it.product(
                it.izip(ay, SV), repeat=2)])
    return 0.5 * result - sum(it.imap(abs, ay))


train = readfile('features.train.txt')
test = readfile('features.test.txt')

print 'Question 18: Consider the Gaussian kernel with gamma=100, and' \
    ' the binary classification problem of `0` versus `not 0`. ' \
    'Consider values of C within {0.001,0.01,0.1,1,10}. Which of the ' \
    'following properties of the soft-margin SVM classifier strictly ' \
    'decreases with those five C?'
Eout = []
Sum_cosi = []
Dist_FSV2HP = []
Num_SV = []
Dual_obj_val = []
for c in C:
    clf = svm.SVC(kernel=kernel, gamma=gamma, C=c)
    clf.fit(*train)
    w = np.array(
        [p * sv for p, sv in it.izip(clf.dual_coef_[0], clf.support_vectors_)])
    w = w.sum(axis=0)
    w_norm = (np.dot(w, w.T))**0.5
    # eout
    Eout.append(1.0 - clf.score(*test))
    # sum of cosi
    cosis = cal_cosi(clf, *train)
    Sum_cosi.append(sum(cosis))
    # the distance of any free SV to the hyperplane
    # in the Z space
    Dist_FSV2HP.append(
        sum([m / w_norm for c, m in it.izip(
            cosis, train[1] * clf.predict(train[0])) if c == 0.0]))
    # the objective value of the dual problem
    Dual_obj_val.append(obj(clf))
    # Number of support vector
    Num_SV.append(sum(clf.n_support_))
P.set_option('display.precision', 6)
print P.DataFrame(
    data=[Eout, Sum_cosi, Dist_FSV2HP, Dual_obj_val, Num_SV],
    index=['eout', 'Sum_cosi', 'Dist_FSV2HP', 'Dual_obj', 'Num_SV'],
    columns=C)
