import matplotlib.pyplot as mpl
import numpy as np
import random


def get_dots(f, a, b, N, A):
	v, step = [[], []], (b - a) / (N - 1)
	v[0] = [a + step * i for i in range(N)]
	v[1] = [f(x) + A * random.uniform(-0.5, 0.5) for x in v[0]]
	return [v[0], v[1]]


def add_dots_to_plot(p, dots):
	p.scatter(dots[0], dots[1])
	return p


def add_fun_to_plot(p, f, lab, a = -4, b = 2, N = 24, mark = ''):
	dots = get_dots(f, a, b, N, 0)
	p.plot(dots[0], [f(x) for x in dots[0]], label = lab, marker = mark)
	return p


def graph(f, w0l, w0r, w1l, w1r, title = 'function'):
	g3d = mpl.figure(figsize = (10, 10)).add_subplot(projection = '3d')
	w0 = np.arange(w0l, w0r, 0.1)
	w1 = np.arange(w1l, w1r, 0.1)
	w0g, w1g = np.meshgrid(w0, w1)
	error_fun = f(w0g, w1g)
	g3d.set_xlabel('d = w0')
	g3d.set_ylabel('c = w1')
	g3d.set_zlabel(title)
	g3d.plot_surface(w0g, w1g, error_fun, cmap = 'plasma')
	mpl.show()
