# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm
import pandas as P

kernel = 'rbf'
gamma = 100.0
c = 0.1
R = [1, 10, 100, 1000, 10000]


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

print 'Question 19: Following Question 18, when fixing C=0.1,' \
    'which of the following values of γ results in the lowest Eout?'
train = readfile('features.train.txt')
test = readfile('features.test.txt')
Eout = []

for r in R:
    clf = svm.SVC(kernel=kernel, gamma=r, C=c)
    clf.fit(*train)
    Eout.append(1.0 - clf.score(*test))

print P.DataFrame(
    data=zip(R, Eout),
    columns=['gamma', 'eout']
    ).sort(columns='eout')
