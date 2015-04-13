import numpy as np
from sklearn import svm
import pandas as P

X = []
Ys = []
digits = []
labels = [0.0, 2.0, 4.0, 6.0, 8.0]
kernel = 'poly'
c = 0.01
degree = 2.0
coef0 = 1.0


with open("features.train.txt", 'r') as file:
    for line in file:
        line = line.decode('utf-8').strip().split(u' ')
        x = map(float, line)
        X.append(x[1:])
        digits.append(x[0])

X = np.array(X)
# content of Y
# ``0'' versus ``not 0''
# ``2'' versus ``not 2''
# ``4'' versus ``not 4''
# ``6'' versus ``not 6''
# ``8'' versus ``not 8''
Ys = [np.array([1.0 if d == l else -1.0 for d in digits]) for l in labels]
Ein = []
a_sums = []
print 'Question 16: Consider the polynomial kernel K(xn,xm)=(1+xTnxm)Q,' \
    'where Q is the degree of the polynomial. With C=0.01, Q=2, which' \
    ' of the following soft-margin SVM classifiers reaches the lowest Ein?'
print 'Question 17: Following Question16, which of the following numbers '\
    'is closest to the maximum summation of alpha within those five' \
    'soft-margin SVM classifiers?'
for l, Y in zip(labels, Ys):
    clf = svm.SVC(kernel=kernel, degree=degree, coef0=coef0, C=c)
    clf.fit(X, Y)
    Ein.append(1.0 - clf.score(X, Y))
    a_sums.append(sum(map(abs, clf.dual_coef_[0])))

print P.DataFrame(
    data=zip(labels, Ein),
    columns=['label', 'e_in']).sort(columns='e_in')
print P.DataFrame(
    data=zip(labels, a_sums),
    columns=['label', 'a_sum']).sort(ascending=False, columns='a_sum')
