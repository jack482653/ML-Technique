import numpy as np
import random

def getdata(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            d = line.strip().split(' ')
            data.append(np.array(map(float, d)))
    return data


def avg(lst):
    return sum(lst) / float(len(lst))


def kmeans_error(MU, S, data):
    result = 0.0
    N = float(sum([len(s) for s in S]))
    for mu, s in zip(MU, S):
        result = result + sum([np.dot(data[d]-mu, data[d]-mu) for d in s])
    return result/N


def np_array_mean(s, data):
    result = 0
    for i in s:
        result = result + data[i]
    return result / float(len(s))


def kmeans_train(k, train_data):
    # step1: initialize mu_1, ...,mu_k
    MU = random.sample(train_data, k)
    # step2: alternating optimization of e_in
    last_S = [set() for _ in range(k)]
    while (True):
        S = [set() for _ in range(k)]
        # step2-1: optimize S_1, ..., S_k
        for i, d in enumerate(train_data):
            distance = [(np.dot(d-mu, d-mu), j) for j, mu in enumerate(MU)]
            min_d, min_j = min(distance, key=lambda x : x[0])
            S[min_j].add(i)
        # step2-2: optimize mu_1, ..., mu_k
        for i, s in enumerate(S):
            MU[i] = np_array_mean(s, train_data)
        # check if convergence
        if last_S == S:
            break
        else:
            last_S = [set([ss for ss in s]) for s in S]
    return MU, S


train = getdata('hw4_kmeans_train.dat')
print 'Experiment with k-Means'
k = 2
err = []
experiment = 500
for _ in range(experiment):
    MU, S = kmeans_train(k, train)
    err.append(kmeans_error(MU, S, train))
print 'k: {} error: {}'.format(k, avg(err))