from prettytable import PrettyTable
import matplotlib.pyplot as plt
import random
import numpy as np


class Being:
	__max_del_per: None
	__borders: None
	__fun: None

	@staticmethod
	def set_max_delta_percentage(del_per):
		Being.__max_del_per = del_per

	@staticmethod
	def set_borders(coordinates):
		Being.__borders = coordinates

	@staticmethod
	def custom_sort_key(being):
		return being.fit()

	@staticmethod
	def set_function(f):
		Being.__fun = f

	@staticmethod
	def show_meta():
		with open('results.txt', 'w') as file:
			file.write('Beings settings so far\n')
			file.write(f'Maximum mutation deviation percent: {Being.__max_del_per}\n')
			file.write(f'Borders of the field are: x from {Being.__borders[0][0]} to {Being.__borders[0][1]}, '
			           f'y from {Being.__borders[1][0]} to {Being.__borders[1][1]}\n')

	def __init__(self):
		self.__chromosomes = [0, 0]

	def rand(self):
		self.__chromosomes = [
			random.uniform(Being.__borders[0][0], Being.__borders[0][1]),
			random.uniform(Being.__borders[1][0], Being.__borders[1][1])]
		return self

	def set(self, x, y):
		self.__chromosomes = [x, y]
		return self

	def set_x(self, x):
		self.__chromosomes[0] = x
		return self

	def set_y(self, y):
		self.__chromosomes[1] = y
		return self

	def get_x(self):
		return self.__chromosomes[0]

	def get_y(self):
		return self.__chromosomes[1]

	def mutate(self, ind):
		delta = Being.__max_del_per * (
				Being.__borders[ind][1] - Being.__borders[ind][0])
		val = random.random() * delta
		if random.random() < 0.5:
			val *= -1
		if self.__chromosomes[ind] + val < Being.__borders[ind][0]:
			self.__chromosomes[ind] = Being.__borders[ind][0]
			return self
		if self.__chromosomes[ind] + val > Being.__borders[ind][1]:
			self.__chromosomes[ind] = Being.__borders[ind][1]
			return self
		self.__chromosomes[ind] += val
		return self

	def fit(self):
		return Being.__fun(self.__chromosomes[0], self.__chromosomes[1])

	def crossing(self, b):
		return [
			Being().set(self.__chromosomes[0], b.__chromosomes[1]),
			Being().set(b.__chromosomes[0], self.__chromosomes[1])
		]


class Generation:
	def __init__(self, num, bs):
		bs.sort(key = Being.custom_sort_key, reverse = True)
		self.__bs = bs
		self.__num = num

	def get_beings(self):
		return self.__bs

	def max_avg_fit(self):
		s = 0
		for b in self.__bs:
			s += b.fit()
		return [self.__bs[0].fit(), s / len(self.__bs)]

	def get_best(self):
		best = self.__bs[0]
		return [best.get_x(), best.get_y(), best.fit()]


def fun(xs, ys):
	return np.exp(-np.power(xs, 2) - np.power(ys, 2))


def generate_beings(num):
	bs = []
	for it in range(num):
		bs.append(Being().rand())
	return bs


def graph(f, xl, xr, yl, yr):
	g3d = plt.figure(figsize = (10, 10)).add_subplot(projection = '3d')
	x = np.arange(xl, xr, 0.1)
	y = np.arange(yl, yr, 0.1)
	xg, yg = np.meshgrid(x, y)
	zg = f(xg, yg)
	g3d.set_xlabel('x')
	g3d.set_ylabel('y')
	g3d.set_zlabel('z')
	g3d.plot_surface(xg, yg, zg, cmap = 'plasma')
	plt.show()


def fit_graph(xy_lists):
	plt.title('Fitness Function Values')
	plt.xlabel('Generation, N')
	plt.ylabel('Function Value')
	plt.plot(xy_lists[0], xy_lists[1], label = 'Average', marker = 'o')
	plt.plot(xy_lists[0], xy_lists[2], label = 'Maximum', marker = '^')
	plt.legend()
	plt.show()


def genetic(num, gens, probability):
	beings = generate_beings(num)
	gen = Generation(0, beings)
	generations = [gen]
	x_values = [0]
	ym_values = [gen.max_avg_fit()[0]]
	y_values = [gen.max_avg_fit()[1]]

	for N in range(1, gens + 1):
		progeny = beings[0].crossing(beings[1])
		progeny.extend(beings[0].crossing(beings[2]))
		beings = progeny

		for b in beings:
			if random.random() < probability:
				b.mutate(0)
			if random.random() < probability:
				b.mutate(1)

		gen = Generation(N, beings)
		generations.append(gen)

		x_values.append(N)
		ym_values.append(gen.max_avg_fit()[0])
		y_values.append(gen.max_avg_fit()[1])

	return [generations, [x_values, y_values, ym_values]]


def show_result(gens):
	ns, xs, ys, fits, max_fits, avg_fits = [], [], [], [], [], []
	skip = ['', '', '', '']
	for gn, gen in enumerate(gens):
		maf = gen.max_avg_fit()
		ns.append(gn)
		max_fits.append(maf[0])
		avg_fits.append(maf[1])
		ns.extend(skip)
		max_fits.extend(skip)
		avg_fits.extend(skip)

		for b in gen.get_beings():
			xs.append(b.get_x())
			ys.append(b.get_y())
			fits.append(b.fit())
		xs.append('')
		ys.append('')
		fits.append('')

	table = PrettyTable()
	titles = ['N', 'X', 'Y', 'FIT', 'MAX FIT', 'AVG FIT']
	table.add_column(titles[0], ns)
	table.add_column(titles[1], xs)
	table.add_column(titles[2], ys)
	table.add_column(titles[3], fits)
	table.add_column(titles[4], max_fits)
	table.add_column(titles[5], avg_fits)

	b = gens[len(gens) - 1].get_best()
	with open('results.txt', 'a') as file:
		file.write(table.get_string())
		file.write(f'\nThe best option so far is: x = {b[0]} y = {b[1]}, fit = {b[2]}\n')


if __name__ == '__main__':
	# setting the initial variables
	x1, y1 = -2, -2
	x2, y2 = 2, 2
	percentage = 5 / 100
	borders = [[x1, x2], [y1, y2]]

	beings_num = 4
	gens_max = 100
	mut_prob = 0.25

	# applying the values
	Being.set_max_delta_percentage(percentage)
	Being.set_borders(borders)
	Being.set_function(fun)

	# genetic algorithm
	result = genetic(beings_num, gens_max, mut_prob)

	# printing the data
	Being.show_meta()
	graph(fun, x1, x2, y1, y2)
	show_result(result[0])
	fit_graph(result[1])
