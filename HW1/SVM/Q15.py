import numpy as np
from sklearn import svm

X = []
Y = []
kernel = 'linear'
c = 0.01
with open("features.train.txt", 'r') as file:
    for line in file:
        line = line.decode('utf-8').strip().split(u' ')
        x = map(float, line)
        X = X + [x[1:]]
        Y = Y + [1.0 if x[0] == 0.0 else -1.0]

X = np.array(X)
Y = np.array(Y)

print 'Question 15: linear kernel, c=0.01, ||w||=?'
clf = svm.SVC(kernel=kernel, C=c)
clf.fit(X, Y)
print [1 for c in clf.dual_coef_[0] if c == 0.0]
w = np.array(
    [p * sv for p, sv in zip(clf.dual_coef_[0], clf.support_vectors_)])
w = w.sum(axis=0)
print "|w|: {}".format((np.dot(w, w.T))**0.5)
