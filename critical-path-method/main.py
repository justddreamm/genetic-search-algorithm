from input import *
from tools import *


def main():
	tutorial()
	g.organize()

	display_edges_table(g.edges(), title = 'Analysed edges', verbose = True)
	display_nodes_table(g.nodes(), title = 'Analysed nodes', verbose = True)

	g.graphs(name = 'v7', save = True)

	print(f'The time of Critical path is {g.cpt()}')
	print(f'There are {g.cpn()} paths with length {g.cpt()}')

	return 0


if __name__ == '__main__':
	main()
