from collections import defaultdict
from itertools import izip, groupby
import random
import sys

def getdata(file_name):
    X = []
    Y = []
    with open(file_name, 'r') as file:
        for line in file:
            d = line.strip().split(' ')
            X.append(map(float, d[:-1]))
            Y.append(float(d[-1]))
    return {'X':X, 'Y': Y}


def avg(lst):
    return sum(lst) / float(len(lst))

def sign(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0


def error_01_u(algo, p, data_set, U):
    error = 0.0
    X = data_set['X']
    Y = data_set['Y']
    #c_e = []
    for x, y, u in izip(X, Y, U):
        if algo(p, x) != y:
            #c_e.append(-1)
            error = error + u
        else:
            pass
            #c_e.append(1)
    return error / len(Y)#, c_e


def decision_stump2(p, x):
    # p = [s, i, t]
    return decision_stump(p[0], x[p[1]], p[2])


def decision_stump(s, xi, t):
    return s * sign(xi - t)


def decision_stump_train(data_set):
    X = data_set['X']
    Y = data_set['Y']
    S = [-1, 1]
    data_size = float(len(X))
    p_list = []
    for index, feature in enumerate(izip(*X)):
        # step1
        f = sorted(feature)
        # step2
        f_mid = [f[0] - 1] # first get negative inf
        f_mid = f_mid + map(lambda x: (x[0] + x[1])/2.0, izip(f, f[1:])) # get mid
        for s in S:
            for t in f_mid:
                p = [s, index, t]
                C = {-1.0: defaultdict(float), 1.0: defaultdict(float)}
                # split to 2 class
                for x,y in izip(X,Y):
                    if decision_stump2(p, x) == -1.0:
                        C[-1.0][y] = C[-1.0][y] + 1.0
                    else:
                        C[1.0][y] = C[1.0][y] + 1.0
                # get gini index
                c1_size = sum(C[-1.0].values())
                c2_size = sum(C[1.0].values())
                impurity_c1 = 1 - sum([(c / c1_size)**2 for c in C[-1.0].values()])
                impurity_c2 = 1 - sum([(c / c2_size)**2 for c in C[1.0].values()])
                
                p_list.append([p, c1_size*impurity_c1 + c2_size*impurity_c2])
    # step3
    p_list = sorted(p_list, key=lambda x : x[1])
    return p_list[0][0]

def same(l):
    return l[1:] == l[:-1]


def most_common(L):
    groups = groupby(sorted(L))
    def _auxfun((item, iterable)):
        return len(list(iterable)), -L.index(item)
    return max(groups, key=_auxfun)[0]


def CRT(train_data, Max_level):
    #node, left, right
    def CT(data, level):
        X = data['X']
        Y = data['Y']
        node = {'g(x)': None, 'b(x)': None, 'left':None, 'right':None}
        # termination criteria met
        if same(X) or same(Y) or level == Max_level:
            node['g(x)'] = most_common(Y)
            return node
        # learn branching criteria b(x)
        node['b(x)'] = decision_stump_train(data)
        # split D to 2 part D1 D2
        D1, D2 = {'X':[],'Y':[]}, {'X':[],'Y':[]}
        for x,y in izip(X,Y):
            if(decision_stump2(node['b(x)'], x) == -1.0):
                D1['X'].append(x)
                D1['Y'].append(y)
            else:
                D2['X'].append(x)
                D2['Y'].append(y)
        # build sub-tree G1, G2
        node['left'] = CT(D1, level + 1)
        node['right'] = CT(D2, level + 1)
        # return G(x)
        return node
    return CT(train_data, 0)


def decision_tree(r, x):
    if r['g(x)'] != None:
        return r['g(x)']
    else:
        if decision_stump2(r['b(x)'], x) == -1.0:
            return decision_tree(r['left'], x)
        else:
            return decision_tree(r['right'], x)

def bootstrap(data, size):
    i = 0
    sample_X = []
    sample_Y = []
    while i < size:
        x = random.choice(data['X'])
        i_x = data['X'].index(x)
        y = data['Y'][i_x]
        sample_X.append(x)
        sample_Y.append(y)
        i = i + 1
    return {'X': sample_X, 'Y': sample_Y}


def random_forest_train(train_data, Max_level):
    Iteration = 300
    Data_size = len(train_data['Y'])
    tree = []
    for _ in range(Iteration):
        sample_data = bootstrap(train_data, Data_size)
        tree.append(CRT(sample_data, Max_level))
    return tree

def random_forest(f, x):
    y = 0.0
    for t in f:
        y = y + decision_tree(t, x)
    return sign(y)

# count = 0
# def count_internal(r):
#     global count
#     if r['b(x)'] != None:
#         count = count + 1
#         count_internal(r['left'])
#         count_internal(r['right'])
#     else:
#         pass

train = getdata('hw3_train.dat')
test = getdata('hw3_test.dat')

# r = CRT(train)
# print '============e_in============'
# U = [1.0 for _ in train['Y']]
# print error_01_u(decision_tree, r, train, U)
# print '============e_out============'
# U = [1.0 for _ in test['Y']]
# print error_01_u(decision_tree, r, test, U)

forest = []
iteration = 100

print 'construct forest:',
sys.stdout.flush()
for i in range(iteration):
    print '{}, '.format(i),
    sys.stdout.flush()
    forest.append(random_forest_train(train, 1))
print ''

# ein = []
# U = [1.0 for _ in train['Y']]
# for trees in forest:
#     for t in trees:
#         ein.append(error_01_u(decision_tree, t, train, U))
# print 'g(t) ein avg:{}'.format(avg(ein))

ein = []
U = [1.0 for _ in train['Y']]
for trees in forest:
    ein.append(error_01_u(random_forest, trees, train, U))
print 'Grf(t) ein avg:{}'.format(avg(ein))

eout = []
U = [1.0 for _ in test['Y']]
for trees in forest:
    eout.append(error_01_u(random_forest, trees, test, U))
print 'Grf(t) ein avg:{}'.format(avg(eout))