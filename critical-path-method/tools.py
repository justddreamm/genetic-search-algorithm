from prettytable import PrettyTable
from sty import *
import networkx as nx
import matplotlib.pyplot as plt

fg.orange = Style(RgbFg(255, 150, 50))


def tutorial():
	print('\nДля удобства отладки для граней (edges) и вершин (nodes) я ввёл систему ID\n')
	print('Пример Node ID')
	print(fg.orange + '\t_N0abd' + fg.rs)
	print('\tN - значит Node')
	print('\t_ - значит нет входящих в вершину работ')
	print('\t0 - уникальный номер вершины')
	print('\tabd - исходящие из вершины работы\n')
	print('Пример Edge ID')
	print(fg.orange + '\t3aE0-1' + fg.rs)
	print('\tE - значит Edge')
	print('\t3 - обозначает длительность выполнения работы')
	print('\ta - уникальное обозначение работы')
	print('\t0 - номер веришины начала')
	print('\t1 - номер веришины конца\n\n')


def display_edges_table(es, title = 'edges', verbose = False):
	table = PrettyTable()
	table.title = fg.orange + title + fg.rs
	if verbose:
		table.add_column(fg.orange + 'edge ID' + fg.rs, [e for e in es])
	else:
		table.add_column(fg.orange + 'edge' + fg.rs, [e.name() for e in es])
	table.add_column(fg.orange + 't' + fg.rs, [e.time() for e in es])
	table.add_column(fg.orange + 't пн' + fg.rs, [e.tls_tee_rfl_rfr()[0] for e in es])
	table.add_column(fg.orange + 't ро' + fg.rs, [e.tls_tee_rfl_rfr()[1] for e in es])
	table.add_column(fg.orange + 'r с' + fg.rs, [e.tls_tee_rfl_rfr()[2] for e in es])
	table.add_column(fg.orange + 'r п' + fg.rs, [e.tls_tee_rfl_rfr()[3] for e in es])
	if verbose:
		table.add_column(fg.orange + 'from' + fg.rs, [e.from_n() for e in es])
		table.add_column(fg.orange + 'to' + fg.rs, [e.to_n() for e in es])
	else:
		table.add_column(fg.orange + 'from' + fg.rs, [e.from_n().name() for e in es])
		table.add_column(fg.orange + 'to' + fg.rs, [e.to_n().name() for e in es])
	print(table)


def display_nodes_table(ns, title = 'nodes', verbose = False):
	table = PrettyTable()
	table.title = fg.orange + title + fg.rs
	if verbose:
		table.add_column(fg.orange + 'node ID' + fg.rs, ns)
	else:
		table.add_column(fg.orange + 'node' + fg.rs, [n.name() for n in ns])
	table.add_column(fg.orange + 'T р' + fg.rs, [n.times()[0] for n in ns])
	table.add_column(fg.orange + 'T п' + fg.rs, [n.times()[1] for n in ns])
	table.add_column(fg.orange + 'R' + fg.rs, [n.times()[2] for n in ns])
	if verbose:
		ins = [[e.__str__() for e in n.ins()] for n in ns]
		outs = [[e.__str__() for e in n.outs()] for n in ns]
	else:
		ins = [[e.name() for e in n.ins()] for n in ns]
		outs = [[e.name() for e in n.outs()] for n in ns]
	table.add_column(fg.orange + 'in edges' + fg.rs, ins)
	table.add_column(fg.orange + 'out edges' + fg.rs, outs)
	print(table)


def show_graph_new(g, moves, i, main_critical = False, save = False, name = ''):
	G = nx.DiGraph()
	for e in g.edges():
		G.add_edge(e.from_n().name(), e.to_n().name(), weight = f'{e.name()} ({e.time()}s)')

	c_edges = []
	r_edges = []
	for e in moves:
		c_edges.append((e.from_n().name(), e.to_n().name()))

	for e in g.edges():
		if (e.from_n().name(), e.to_n().name()) not in c_edges:
			r_edges.append((e.from_n().name(), e.to_n().name()))

	pos = nx.circular_layout(G)
	nx.draw_networkx_nodes(G, pos, cmap = plt.get_cmap('jet'), node_size = 200)
	nx.draw_networkx_edges(G, pos, edgelist = c_edges, edge_color = 'r', arrows = True, label = f'Critical path {i}')
	nx.draw_networkx_edges(G, pos, edgelist = r_edges, arrows = True, alpha = 0.7, )
	nx.draw_networkx_edge_labels(G, pos, font_size = 10, edge_labels = nx.get_edge_attributes(G, 'weight'),
		label_pos = 0.3)
	nx.draw_networkx_labels(G, pos)
	if main_critical:
		plt.title(f'Critical path {i} through all R = 0 events')
	else:
		plt.title(f'Critical path {i}')
	if save:
		plt.savefig(f'{name}-critical-path-{i}.png')
	plt.show()
