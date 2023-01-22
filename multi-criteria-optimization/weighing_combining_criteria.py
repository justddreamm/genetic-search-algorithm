from tools import show_matrix, show_table, normalize_matrix, norm_list
from termcolor import colored
import numpy as np
from sty import fg


def pairwise_comparison(i, j, mci):
	if mci[i] > mci[j]:
		return 1
	elif mci[i] < mci[j]:
		return 0
	else:
		return 0.5


def make_weights_vector(criteria, mci):
	alpha = []
	for i in range(len(criteria)):
		buffer = []
		for j in range(len(criteria)):
			if not i == j:
				buffer.append(pairwise_comparison(i, j, mci))
		alpha.append(sum(buffer))
	return alpha


def matrix_vector_multiply(m, v):
	vec = []
	for line in m:
		buffer = []
		for i, el in enumerate(line):
			buffer.append(el * v[i])
		vec.append(sum(buffer))
	return vec


def weigh_combine_criteria(w, c, aim_max = True):
	ws = make_weights_vector([it[0] for it in w], [it[1] for it in w])
	av = norm_list(ws)
	normed = normalize_matrix(show_table(w, c, show = False))
	print(fg.green + 'Normalized matrix' + fg.rs)
	show_matrix(normed, w, c)
	fv = matrix_vector_multiply(normed, av)
	print('Normalized criteria priority weights vector: ' + fg.blue + f'{av}' + fg.rs)
	print('After multiplying matrix by a vector: ' + fg.red + f'{fv}' + fg.rs)
	if aim_max:
		print('We ' + fg.green + 'maximize' + fg.rs + ' the parameters')
		ind = np.array(fv).argmax()
	else:
		print('We ' + fg.green + 'minimize' + fg.rs + ' the parameters')
		ind = np.array(fv).argmin()
	print(f'The most suitable option is ' + colored(f'{c[ind].__str__()}', 'red') + f' {c[ind].ws()}')
