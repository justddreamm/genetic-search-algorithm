from gsl import *
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def fun(x, y):
	return np.exp(-np.power(x, 2) - np.power(y, 2))


def graph(f, xl, xr, yl, yr):
	g3d = plt.figure(figsize = (10, 10)).add_subplot(projection = '3d')
	g3d.set_xlabel('x')
	g3d.set_ylabel('y')
	g3d.set_zlabel('z')
	x = np.arange(xl, xr, 0.1)
	y = np.arange(yl, yr, 0.1)
	xg, yg = np.meshgrid(x, y)
	zg = f(xg, yg)
	g3d.plot_surface(xg, yg, zg, cmap = 'plasma')
	plt.show()


def fit_graph(xy_lists, g0):
	plt.title('fitness function')
	plt.xlabel('generation, n')
	plt.ylabel('function value')
	plt.plot(xy_lists[0], xy_lists[1], label = 'average', marker = 'o')
	plt.plot(xy_lists[0], xy_lists[2], label = 'maximum', marker = '^')
	plt.scatter(g0.n(), g0.best_fit(), label = 'best being', s = 100, color = 'red')
	plt.legend()
	plt.show()


def show_result(g0, gens):
	skip = [''] * len(gens[0].beings())
	values = {
		'n': [],
		'x': [],
		'y': [],
		'fits': [],
		'max_fits': [],
		'avg_fits': []
	}

	for gen in gens:
		maf = gen.max_avg_fit()
		values['n'].append(gen.n())
		values['n'].extend(skip)
		values['max_fits'].append(maf[0])
		values['max_fits'].extend(skip)
		values['avg_fits'].append(maf[1])
		values['avg_fits'].extend(skip)

		for b in gen.beings():
			values['x'].append(b.get(0))
			values['y'].append(b.get(1))
			values['fits'].append(b.fit())

		values['x'].append(skip[0])
		values['y'].append(skip[0])
		values['fits'].append(skip[0])

	table = PrettyTable()
	titles = ['N', 'X', 'Y', 'FIT', 'MAX FIT', 'AVG FIT']
	for i, key in enumerate(values):
		table.add_column(titles[i], values[key])

	b0 = g0.best_values()
	with open('results.txt', 'a') as file:
		file.write(table.get_string())
		file.write(f'\nThe best option so far is: x = {b0[0]} y = {b0[1]}, fit = {b0[2]}\n')


def main():
	# setting the initial variables
	x1, y1 = -2, -2
	x2, y2 = 2, 2
	percentage = 5 / 100
	borders = [[x1, x2], [y1, y2]]
	beings_num = 4
	gens_max = 60
	mut_prob = 0.25

	# applying the values
	Being.set_dev_amp(percentage)
	Being.set_mut_prob(mut_prob)
	Being.set_borders(borders)
	Being.set_fun(fun)

	# genetic algorithm
	result = genetic(beings_num, gens_max, mut_prob)

	# printing the data
	Being.show_meta()
	graph(fun, x1, x2, y1, y2)
	show_result(result[0], result[1])
	fit_graph(result[2], result[0])


if __name__ == '__main__':
	main()
