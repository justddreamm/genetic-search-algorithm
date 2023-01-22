from tools import show_table
from matplotlib import pyplot as plt
from termcolor import colored


def pareto(w, c, main_criteria, aim_max = True):
	c_indexes = [[it[0] for it in w].index(c) for c in main_criteria]
	plt.title(f'Pareto Set Method')
	show_table(w, c)
	print('The optimized criteria: ' + colored(f'{main_criteria}', 'green'))
	plt.xlabel(w[c_indexes[0]][0])
	plt.ylabel(w[c_indexes[1]][0])
	can_values = []
	for i, can in enumerate(c):
		can_values.append([can.ws()[c_indexes[0]], can.ws()[c_indexes[1]]])
		plt.scatter(can.ws()[c_indexes[0]], can.ws()[c_indexes[1]], label = can.__str__())
	if aim_max:
		utopia = [max(can.ws()[c_indexes[0]] for can in c), max(can.ws()[c_indexes[1]] for can in c)]
	else:
		utopia = [min(can.ws()[c_indexes[0]] for can in c), min(can.ws()[c_indexes[1]] for can in c)]
	plt.scatter(utopia[0], utopia[1], label = 'utopia')
	distances = [((cv[0] - utopia[0]) ** 2 + (cv[1] - utopia[1]) ** 2) ** 0.5 for cv in can_values]
	plt.legend()
	plt.show()
	best = c[distances.index(min(distances))]
	print(f'The most suitable option is ' + colored(f'{best.__str__()}', 'red') + f' {best.ws()}')
