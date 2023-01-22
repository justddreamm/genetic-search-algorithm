from prettytable import PrettyTable
import matplotlib.pyplot as mpl
from input import *
import random


def f(x):
	return math.sin(x) + 0.5


Xk = [k * (X_MAX - X_MIN) / K for k in range(K + 1)]
Qk = [random.uniform(-A, A) for k in range(K + 1)]
F = [f(X_MIN + k * (X_MAX - X_MIN) / K) for k in range(K + 1)]
FN = [F[k] + Qk[k] for k in range(K + 1)]
LAMBDAS = [l / L for l in range(L + 1)]


def m_from(r):
	return int((r - 1) / 2)


def w(a):
	fs = filtered_signal_list(a)
	return sum([(fs[k] - fs[k - 1]) ** 2 for k in range(1, K + 1)]) ** 0.5


def delta(a):
	fs = filtered_signal_list(a)
	return (sum([(fs[k] - FN[k]) ** 2 for k in range(K + 1)]) / K) ** 0.5


def rand_a(r):
	M = m_from(r)
	a = [random.random() for i in range(r)]
	a[M] = random.uniform(0, 1)
	for m in range(2, M + 1):
		val = 0.5 * random.uniform(0, 0.1 - sum([a[s - 1] for s in range(m + 1, r - m + 1)]))
		a[m - 1] = val
		a[r - m] = val
	val = 0.5 * (1 - sum([a[s - 1] for s in range(2, r)]))
	a[0] = val
	a[r - 1] = val
	return a


def dist(a):
	return (w(a) ** 2 + delta(a) ** 2) ** 0.5


def filtered_signal_list(a):
	M = m_from(len(a))
	fs = [0] * (K + 1)
	for k in range(M, K - M + 1):
		fs[k] = sum([(a[j + M - k] / FN[j]) for j in range(k - M, k + M + 1)]) ** (-1)
	return fs


def j(a, lam):
	return lam * w(a) + (1 - lam) * delta(a)


def passive_search(lambdas, r, e = 0.01):
	return [random_search(P, r, lam, e = e) for lam in lambdas]


def random_search(p, r, lam, e = 0.01):
	n = math.ceil(math.log(1 - p) / math.log(1 - e / (X_MAX - X_MIN)))
	best = [None, None, None, None, None]
	for it in range(n):
		a = rand_a(r)
		if it == 0:
			best = [lam, dist(a), a, w(a), delta(a), j(a, lam)]
		elif j(a, lam) < best[5]:
			best = [lam, dist(a), a, w(a), delta(a), j(a, lam)]
	return best


def add_fun_to_plot(p, x, y, lab = 'function', mark = ''):
	p.plot(x, y, label = lab, marker = mark)
	return p


def show_table(title, labs, results):
	table = PrettyTable()
	table.title = title
	for i in range(6):
		table.add_column(labs[i], [item[i] for item in results])
	print(table)


def stochastic_filtering(r, e = E):
	results = passive_search(LAMBDAS, r, e = e)

	mpl.title(f'Dots for R = {r}')
	mpl.xlabel('w')
	mpl.ylabel('delta')
	for i in range(len(results[3])):
		mpl.scatter([item[3] for item in results], [item[4] for item in results])
	mpl.show()

	labels = ['h', 'dist', 'alpha', 'w', 'delta', 'J']
	show_table(f'Stochastic filtering with Euclidean metric for R = {r}, e = {e}',
		labels, results)
	results.sort(key = lambda x: x[1])
	show_table(f'Results with minimum distance for R = {r}, e = {e}',
		labels, [item for item in [results[0]]])

	filtered = filtered_signal_list(results[0][2])
	mpl.title(f'Functions for R = {r}')
	mpl.plot(Xk, F, label = 'f(x) = sin(x) + 0.5')
	mpl.plot(Xk, FN, label = 'noisy')
	mpl.plot(Xk, filtered, label = 'filtered')
	mpl.legend()
	mpl.show()

	return 0
