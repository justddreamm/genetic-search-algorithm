from tools import show_table
from termcolor import colored


def check_candidate(line, restrictions, oci, lower_border = True):
	for i, it in enumerate(restrictions):
		if oci != i:
			if lower_border:
				if line[i] < it:
					return False
			elif line[i] > it:
				return False
	return True


def criteria_constraint(w, c, main_criteria, aim_max = True):
	oci = [t[0] for t in w].index(main_criteria)
	print('The optimized criteria: ' + colored(f'{main_criteria}', 'green'))
	m = show_table(w, c)
	min_allowed = [0.5 * max(line[i] for line in m) for i in range(len(w))]
	min_allowed[oci] = None
	print(f'The lowest values border: {min_allowed}')
	pass_cs = [c[i] for i, line in enumerate(m) if check_candidate(line, min_allowed, oci, lower_border = aim_max)]

	def sort_key(can):
		return can.ws()[oci]

	pass_cs.sort(key = sort_key, reverse = aim_max)
	print(f'Candidates to meet minimum requirements: {[p.__str__() for p in pass_cs]}')
	print('The most suitable option is ' + colored(f'{pass_cs[0]}', 'red') + f' {pass_cs[0].ws()}')
	return pass_cs[0]
