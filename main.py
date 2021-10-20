import matplotlib.pyplot as plt
import random
import numpy as np


class Being:
	def __init__(self):
		self.x = 0
		self.y = 0

	def rand(self, xl, xr, yl, yr):
		self.x = random.uniform(xl, xr)
		self.y = random.uniform(yl, yr)
		return self

	def set(self, x, y):
		self.x = x
		self.y = y
		return self

	def rand_x(self, xl, xr):
		self.x = random.uniform(xl, xr)
		return self

	def rand_y(self, yl, yr):
		self.y = random.uniform(yl, yr)
		return self

	def fit(self, f):
		return f(self.x, self.y)


class Iteration:
	def __init__(self, N, bs, f):
		self.N = N
		self.bs = bs
		self.f = f

	def __str__(self):
		s = f'{self.N}\t'
		if len(self.bs) > 0:
			s += f'\t\t{self.bs[0].x}'
			s += f'\t\t\t{self.bs[0].y}'
			# s += f'\t\t\t{Being.fit(self.bs[0], self.f)}'
			s += f'\t\t\t{self.bs[0].fit(self.f)}'

			val = max_avg_fit(self.bs, self.f)
			s += f'\t\t\t{val[0]}\t\t\t{val[1]}\n'
			for it in range(1, len(self.bs)):
				s += f'\t\t\t{self.bs[it].x}'
				s += f'\t\t\t{self.bs[it].y}'
				# s += f'\t\t\t{Being.fit(self.bs[it], self.f)}\n'
				s += f'\t\t\t{self.bs[it].fit(self.f)}\n'

		return s

	def get_best_xyz_list(self):
		self.bs.sort(key = custom_sort_key, reverse = True)
		best = self.bs[0]
		return [best.x, best.y, best.fit(self.f)]


def fun(xs, ys):
	return np.exp(-np.power(xs, 2) - np.power(ys, 2))


def custom_sort_key(being):
	return being.fit(fun)


def generate_beings(num, xl, xr, yl, yr):
	bs = []
	for it in range(num):
		bs.append(Being().rand(xl, xr, yl, yr))
	return bs


def max_avg_fit(bs, f):
	s = 0
	best = 0
	for b in bs:
		s += b.fit(f)
		if b.fit(f) > best:
			best = b.fit(f)
	return [best, s / len(bs)]


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


def genetic(f, xl, xr, yl, yr, num, gens):
	graph(f, xl, xr, yl, yr)

	beings = generate_beings(num, xl, xr, yl, yr)
	beings.sort(key = custom_sort_key, reverse = True)
	iterations = [Iteration(0, beings, f)]
	x_values = [0]
	ym_values = [max_avg_fit(beings, f)[0]]
	y_values = [max_avg_fit(beings, f)[1]]

	for N in range(1, gens + 1):
		beings = [
			Being().set(beings[0].x, beings[1].y),
			Being().set(beings[1].x, beings[0].y),
			Being().set(beings[0].x, beings[2].y),
			Being().set(beings[2].x, beings[0].y),
		]

		# series of experiments showed that we should take into consideration
		# whether the first coordinate mutated whilst judging if another one
		# should be mutated as well. if one is mutated already, then the second
		# one has its chance for this reduced

		probability = 0.25
		for b in beings:
			# this seems to be better due to the tests conducted
			# if random.random() < probability:
			# 	mutated_x = True
			# 	if random.random() < 0.5:
			# 		b.rand_x(xl, xr)
			# 	else:
			# 		b.rand_y(yl, yr)
			# 		mutated_x = False
			# 	if random.random() < probability:
			# 		if mutated_x:
			# 			b.rand_y(yl, yr)
			# 		else:
			# 			b.rand_x(xl, xr)

			if random.random() < probability:
				b.rand_x(xl, xr)
			if random.random() < probability:
				b.rand_y(yl, yr)

		beings.sort(key = custom_sort_key, reverse = True)
		iterations.append(Iteration(N, beings, f))

		if N < 10 or N % 10 == 0:
			x_values.append(N)
			ym_values.append(max_avg_fit(beings, f)[0])
			y_values.append(max_avg_fit(beings, f)[1])

	return [iterations, [x_values, y_values, ym_values]]


def show_result(its):
	print('N\t\t\t\tX\t\t\t\t\tY\t\t\t\t\t\t\tFIT\t\t\t\t\t\tMAX '
	      'FIT\t\t\t\t\tAVG FIT\n')
	for num, it in enumerate(its):
		if num < 10 or num % 10 == 0:
			print(it)
	b = its[len(its) - 1].get_best_xyz_list()
	print(f'The best option so far is: x = {b[0]} y = {b[1]}, fit = {b[2]}\n')


if __name__ == '__main__':
	# setting the initial variables
	x1 = -2
	x2 = 2
	y1 = -2
	y2 = 2
	n = 4
	gen_m = 100

	# the very function
	result = genetic(fun, x1, x2, y1, y2, n, gen_m)
	show_result(result[0])
	fit_graph(result[1])
