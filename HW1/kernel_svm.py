from cvxopt import matrix, spmatrix, solvers

def kernel2(vec_xn, vec_xm):
    return (1 + vec_xn.T * vec_xm)**2

x = [
        matrix([1,0], (2,1)),
        matrix([0,1], (2,1)),
        matrix([0,-1], (2,1)),
        matrix([-1,0], (2,1)),
        matrix([0,2], (2,1)),
        matrix([0,-2], (2,1)),
        matrix([-2,0], (2,1))]
y = [-1, -1, -1, 1, 1, 1, 1]
P = matrix([y[n] * y[m] * kernel2(x[n], x[m]) for n in range(len(y)) for m in range(len(y))], (len(y), len(y)))
vec_q = matrix(-1.0, (len(y), 1))
A = matrix([matrix(y).T, -1 * matrix(y).T])
G = spmatrix(-1.0, range(len(y)), range(len(y)))
G = matrix([A, G])
vec_h = matrix(0.0, (len(y) + 2, 1))

sol = solvers.qp(P, vec_q, G, vec_h)

b = [y[i] - sum([c * a * kernel2(b, x[i]) for a, b, c in zip(sol['x'], x, y)]) for i in range(7)]
for l in b:
	print l