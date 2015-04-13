from itertools import izip, imap
import math

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


def error_01_u(algo, p, data_set, u):
    error = 0.0
    X = data_set['X']
    Y = data_set['Y']
    c_e = []
    for d in izip(X, Y, u):
        if algo(p, d[0]) != d[1]:
            c_e.append(-1)
            error = error + d[2]
        else:
            c_e.append(1)
    return error / len(Y), c_e


def decision_stump2(p, x):
    # p = [s, i, t]
    return decision_stump(p[0], x[p[1]], p[2])

def decision_stump(s, xi, t):
    return s * sign(xi - t)


def decision_stump_train(data_set, u):
    X = data_set['X']
    Y = data_set['Y']
    S = [-1, 1]
    p_list = []
    for index, feature in enumerate(izip(*X)):
        # step1
        f = sorted(feature)
        # step2
        f_mid = [f[0] - 1] # first get negative inf
        f_mid = f_mid + map(lambda x: (x[0] + x[1])/2.0, izip(f, f[1:])) # get mid
        bset_ein = 1000.0 # find best s, t for xi
        best_s = None
        best_t = None
        for s in S:
            for t in f_mid:
                p = [s, index, t]
                ein, _ = error_01_u(decision_stump2, p, data_set, u)
                if ein < bset_ein:
                    bset_ein = ein
                    best_s = s
                    best_t = t
        p_list.append([best_s, index, best_t, bset_ein])
    # step3
    p_list = sorted(p_list, key=lambda x : x[-1])
    return p_list[0]


def ada_boost(data_set):
    # initialize u
    data_size = len(data_set['Y'])
    u = [1.0 / data_size for _ in range(data_size)]
    T = 300 # iteration
    a = []
    e_min = 10000
    p_list = []
    for i in range(T):
        print 'U{}: {} '.format(i+1, sum(u)),
        p = decision_stump_train(data_set, u)
        _, c_e = error_01_u(decision_stump2, p[:-1], data_set, u)
        incorrect = p[3] * float(data_size)
        e = incorrect / sum(u)
        if e < e_min:
            e_min = e
        t = ((1.0 - e) / e) ** 0.5
        for i in range(data_size):
            if c_e[i] == 1: #correct
                u[i] = u[i] / t
            else:
                u[i] = u[i] * t
        p_list.append(p)
        a.append(math.log1p(t))
    print 'e_min: {}'.format(e_min)
    return p_list, a

def ada_g(p, x):
    p_list = p[0]
    a_list = p[1]
    y = 0.0
    for p, a in izip(p_list, a_list):
        y = y + a * decision_stump2(p[:-1], x)
    return sign(y)

train = getdata('hw2_adaboost_train.dat')
test = getdata('hw2_adaboost_test.dat')
train_size = len(train['Y'])
u1 = [1.0 / float(train_size) for _ in range(train_size)]
u2 = [1.0 for _ in range(train_size)]
print '============Q12-13============'
p_star = decision_stump_train(train, u1)
print error_01_u(decision_stump2, p_star[:-1], train, u2)
p_list, a_list = ada_boost(train)
print error_01_u(ada_g, [p_list, a_list], train, u2)
print '============e_out============'
test_size = len(test['Y'])
u2 = [1.0 for _ in range(test_size)]
print error_01_u(decision_stump2, p_list[0][:-1], test, u2)
print error_01_u(ada_g, [p_list, a_list], test, u2)