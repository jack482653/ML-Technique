# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm, cross_validation
import pandas as P

kernel = 'rbf'
gamma = 100.0
c = 0.1
R = [1, 10, 100, 1000, 10000]
iteration = 100


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


def avg(l):
    return sum(l)/float(len(l))


print 'Question 20: Following Question 18 and consider a validation' \
    ' procedure that randomly samples 1000 examples from the training set' \
    ' for validation and leaves the other examples for training g−SVM. ' \
    'Fix C=0.1 and use the validation procedure to choose the best γ among' \
    ' {1,10,100,1000,10000} according to Eval. If there is a tie of Eval, ' \
    'choose the smallest γ. Repeat the procedure 100 times. Which of the ' \
    'following values of γ is selected the most number of times?'
train = readfile('features.train.txt')
Eavg_val = []

for r in R:
    clf = svm.SVC(kernel=kernel, gamma=r, C=c)
    E_val = []
    for _ in xrange(iteration):
        train_X, Val_X, train_Y, Val_Y = cross_validation.train_test_split(
            *train, test_size=1000)
        clf.fit(train_X, train_Y)
        E_val.append(1.0 - clf.score(Val_X, Val_Y))
    Eavg_val.append(avg(E_val))

print P.DataFrame(
    data=zip(R, Eavg_val),
    columns=['gamma', 'e_val']).sort(columns='e_val')
