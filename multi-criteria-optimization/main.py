from criteria_constraint import criteria_constraint
from pareto import pareto
from weighing_combining_criteria import weigh_combine_criteria
from hierarchy_analysis import hierarchy_analysis
from input import W, CANDIDATES
from sty import bg


def main():
	print(bg.blue + 'Criteria Constraint Method:' + bg.rs)
	criteria_constraint(W, CANDIDATES, 'character', aim_max = True)
	print()

	print(bg.blue + 'Pareto Set Method:' + bg.rs)
	pareto(W, CANDIDATES, ['character', 'appearance'], aim_max = True)
	print()

	print(bg.blue + 'Weighing and Combining Criteria Method:' + bg.rs)
	weigh_combine_criteria(W, CANDIDATES, aim_max = True)
	print()

	print(bg.blue + 'Hierarchy Analysis Method:' + bg.rs)
	hierarchy_analysis(W, CANDIDATES, aim_max = True)
	print()


if __name__ == '__main__':
	main()
