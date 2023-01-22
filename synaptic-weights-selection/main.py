from gsl import *
import plots as plt
import matplotlib.pyplot as mpl

D, C = 0, 8


def f_frame(x, ws):
	return ws[0] + x * ws[1]


def f_by_weights_f(ff, ws):
	def fun_by_weights(x):
		return ff(x, ws)
	return fun_by_weights


def f(x):
	return f_by_weights_f(f_frame, [D, C])(x)


def f_error_to_f_by_weights(dots):
	def f_to_return(ws):
		fun_predicted = f_by_weights_f(f_frame, ws)
		return sum((fun_predicted(x) - dots[1][i]) ** 2 for i, x in enumerate(dots[0]))
	return f_to_return


def gradient(dots):
	c = (len(dots[0]) * sum(x * dots[1][i] for i, x in enumerate(dots[0])) - sum(dots[0]) * sum(dots[1])) / (
			len(dots[0]) * sum(x ** 2 for x in dots[0]) - sum(dots[0]) ** 2)
	return [(sum(dots[1]) - c * sum(dots[0])) / len(dots[0]), c]


def f_args_from_list_to_positional(fun_to_modify):
	def f_return(w0, w1):
		return fun_to_modify([w0, w1])
	return f_return


def conduct_experiment(init_f, a, b, N, A, pop, gens, mp, max_flag = False):
	dots = plt.get_dots(init_f, a, b, N, A)
	error_fun = f_args_from_list_to_positional(f_error_to_f_by_weights(dots))

	Being.set_mut_prob(mp)
	Being.set_dev_amp(0.05)  # let it be as a default
	x1, x2 = -20, 20
	y1, y2 = -20, 20
	Being.set_borders([[x1, x2], [y1, y2]])  # pretty wide range
	Being.set_f(error_fun)
	ws = genetic(pop, gens, maximum = max_flag)[0].best_being().values()[:2]

	coefficients = gradient(dots)
	mpl.title(f'plots w/ A = {A}')
	plt.add_fun_to_plot(mpl, init_f, 'given function')
	plt.add_fun_to_plot(mpl, f_by_weights_f(f_frame, ws), 'genetic function')
	plt.add_fun_to_plot(mpl, f_by_weights_f(f_frame, coefficients), 'gradient function')
	plt.add_dots_to_plot(mpl, dots)
	mpl.legend()
	mpl.show()

	with open('results.txt', 'a') as file:
		file.write(f'experiment w/ A = {A}\n')
		file.write(f'genetic coefficients: {ws}\n')
		file.write(f'lowest squares method coefficients: {coefficients}\n\n')

	plt.graph(error_fun, x1, x2, y1, y2, f'err-fun-sur w/ A ={A}')
	return [ws, dots, coefficients]


def main():
	a, b = -4, 2
	N = 24
	A1, A2 = 10, 0

	pop = 4
	gens = 50
	mut_prob = 0.25

	with open('results.txt', 'w') as file:
		file.write(f'data given:\nd = w0 = {D}\nc = w1 = {C}\n\n')

	conduct_experiment(f, a, b, N, A1, pop, gens, mut_prob, False)
	conduct_experiment(f, a, b, N, A2, pop, gens, mut_prob, False)


if __name__ == '__main__':
	main()
