from prettytable import PrettyTable
from tools import show_table, norm_list
import numpy as np
from sty import *


def make_pairwise_comparison_matrix(vec, name):
	m = [[el1[1] / el2[1] for el2 in vec] for el1 in vec]
	table = PrettyTable()
	table.add_column('', [el[0] for el in vec])
	for i, can in enumerate(vec):
		table.add_column(can[0], [line[i] for line in m])
	bg.orange = Style(RgbBg(255, 150, 50))
	fg.orange = Style(RgbFg(255, 150, 50))
	print(bg.orange + fg.black + f'{name} table' + fg.rs + bg.rs)
	sum_vec = [sum(line) for line in m]
	table.add_column(fg.orange + 'Sum line' + fg.rs, sum_vec)
	nl = norm_list(sum_vec)
	table.add_column(fg.orange + 'Normalized sum line' + fg.rs, nl)
	print(table)
	return nl


def hierarchy_analysis(w, c, aim_max = True):
	print(fg.green + 'Main table' + fg.rs)
	mat = show_table(w, c)
	m = []
	for ind in range(len(w)):
		col = make_pairwise_comparison_matrix([[c[i].__str__(), l[ind]] for i, l in enumerate(mat)], w[ind][0])
		m.append(col)
	mul_col = make_pairwise_comparison_matrix(w, 'Weights priority assessment')
	fv = [sum([m[i][k] * el for i, el in enumerate(mul_col)]) for k in range(len(m[0]))]
	print('After multiplying the vector is ' + fg.orange + f'{fv}' + fg.rs)
	if aim_max:
		print('We ' + fg.green + 'maximize' + fg.rs + ' the parameters')
		ind = np.array(fv).argmax()
	else:
		print('We ' + fg.green + 'minimize' + fg.rs + ' the parameters')
		ind = np.array(fv).argmin()
	print(f'The most suitable option is ' + fg.green + f'{c[ind].__str__()}' + f' {c[ind].ws()}' + fg.rs)
