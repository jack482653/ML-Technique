import numpy as np
from sklearn import svm

c = [0.001, 0.01, 0.1, 1, 10]
r = 100

# training data
X = []
Y = []
with open("features.train.txt",'r') as file:
    for line in file:
    	line = line.decode('utf-8').strip().split(u' ')
    	x = [float(a) for a in line]
    	X = X + [x]

X = np.array(X)
# ``0'' versus ``not 0''
Y = np.array([1.0 if x[0] == 0.0 else -1.0 for x in X])

# test data
XX = []
YY = []
with open("features.test.txt",'r') as file:
    for line in file:
        line = line.decode('utf-8').strip().split(u' ')
        x = [float(a) for a in line]
        XX = XX + [x]

XX = np.array(XX)
# ``0'' versus ``not 0''
YY = np.array([1.0 if x[0] == 0.0 else -1.0 for x in XX])

def kernel(a, b):
    return np.exp(-r * (np.linalg.norm(a - b) ** 2))


for cc in c:
    clf = svm.SVC(kernel='rbf', gamma = r, C = cc)
    clf.fit(X, Y)
    # Eout
    # print 'Acc: {}'.format(clf.score(XX, YY))

    # number of support vectors
    # print '# support vectors: {}'.format(sum(clf.n_support_))

    # the objective value of the dual problem
    # summation = 0.5 * sum([a1 * a2 * kernel(X[i1], X[i2]) for a1, i1 in zip(clf.dual_coef_[0], clf.support_) for a2, i2 in zip(clf.dual_coef_[0], clf.support_)])
    # summation = summation - sum([abs(a) for a in clf.dual_coef_[0]])
    # print 'Sum: {}'.format(summation)

    # soci
    # SV_x = [X[i] for i in clf.support_]
    # SV_y = [Y[i] for i in clf.support_]
    # score = []
    # for a, df in zip(clf.dual_coef_[0], clf.decision_function(SV_x)):
    #     df = df[0]
    #     if a * df > 1:
    #         score = score + [0]
    #     else:
    #         score = score + [1 - a * df]
    # print 'soci: {}'.format(sum(score))

    # the distance of any free support vector to the hyperplane in the (infinite-dimensional) Z space
    SV_x = [X[i] for i in clf.support_]
    SV_y = [Y[i] for i in clf.support_]
    for a, sx, sy in zip(clf.dual_coef_[0], SV_x, SV_y):
        if abs(a) < cc:
            b = sy - sum([aa * kernel(x, sx) for aa, x in zip(clf.dual_coef_[0], SV_x)])
            print 'b: {}'.format(b)
            break




# w = np.array([a * X[clf.support_[i]] for i, a in enumerate(clf.dual_coef_[0])]).sum(axis=0)
# print "|w|: {}".format(sum([i**2 for i in w])**0.5)
# print len(clf.dual_coef_[0]), len(clf.support_vectors_)