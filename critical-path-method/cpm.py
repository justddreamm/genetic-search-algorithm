import copy

import numpy as np
from tools import *


class Node:
	__nc = 0

	@staticmethod
	def can_merge_ns(n1, n2):
		if id(n1) == id(n2):
			return False
		if len(n1.ins()) != len(n2.ins()):
			return False
		for b in n1.ins():
			if b not in n2.ins():
				return False
		return True

	@staticmethod
	def critical_path(n):
		if n.outs():
			periods = [Node.critical_path(e.to_n())[0] + e.time() if e.to_n().is_critical() else 0 for e in n.outs()]
			return [max(periods, default = 0), n.outs()[np.array(periods).argmax()]]
		else:
			return [0, None]

	def __init__(self):
		Node.__nc += 1
		self.__id = Node.__nc
		self.__ins = []
		self.__outs = []
		self.__telr = [None, None, None]
		self.__is_critical = False
		self.__is_start = None
		self.__is_end = None

	def reset_parameters(self):
		self.__telr = [None, None, None]

	def check_if_is_start(self):
		if not self.__ins:
			self.__is_start = True
		else:
			self.__is_start = False

	def delete_in(self, ei):
		for e in self.__ins:
			if id(ei) == id(e):
				self.__ins.remove(e)

	def delete_out(self, eo):
		for e in self.__outs:
			if id(eo) == id(e):
				self.__outs.remove(e)

	def check_if_is_end(self):
		if not self.__outs:
			self.__is_end = True
		else:
			self.__is_end = False

	def is_end(self):
		return self.__is_end

	def is_start(self):
		return self.__is_start

	def is_critical(self):
		return self.__is_critical

	def update(self):
		self.check_if_is_start()
		self.check_if_is_end()
		return self

	def set_outs(self, es):
		self.__outs = es
		for b in self.__outs:
			b.set_from_n(self)
		self.update()
		return self

	def set_ins(self, es):
		self.__ins = es
		for b in self.__ins:
			b.set_to_n(self)
		self.update()
		return self

	def update_id(self, id):
		self.__id = id
		for e in self.__ins:
			e.set_to_n(self)
		for e in self.__outs:
			e.set_from_n(self)
		return self

	def calculate_te(self):
		self.__telr[0] = max([ei.from_n().calculate_te() + ei.time() for ei in self.ins()], default = 0)
		return self.__telr[0]

	def calculate_tl(self):
		self.__telr[1] = min([eo.to_n().calculate_tl() - eo.time() for eo in self.outs()], default = self.__telr[0])
		return self.__telr[1]

	def calculate_r(self):
		self.__telr[2] = self.__telr[1] - self.__telr[0]
		if self.__telr[2] == 0:
			self.__is_critical = True

	def merge(self, n):
		for e in n.outs():
			if e not in self.__outs:
				self.__outs.append(e)
		for b in n.ins():
			b.set_to_n(self)
		for b in n.outs():
			b.set_from_n(self)
		self.update()
		return self

	def ins(self):
		return self.__ins

	def outs(self):
		return self.__outs

	def times(self):
		return self.__telr

	def name(self):
		return self.__id

	def longest_path_to_end(self, g, aim, time_start, prev_edges):
		if self.outs():
			for eo in self.outs():
				h = copy.deepcopy(prev_edges)
				h.append(eo)
				eo.to_n().longest_path_to_end(g, aim, time_start + eo.time(), h)
		else:
			if time_start >= aim and not g.already_has_cp(prev_edges):
				g.add_cp(prev_edges)

	def __str__(self):
		pl = ''.join([f'{fe.name()}' for fe in self.ins()])
		ol = ''.join([f'{te.name()}' for te in self.outs()])
		pl = '_' if not pl else pl
		ol = '_' if not ol else ol
		return pl + f'N{self.name()}' + ol


class Edge:
	__ec = 0

	def __init__(self, n, t):
		Edge.__ec += 1
		self.__id = Edge.__ec
		self.__n = n
		self.__t = t
		self.__t_ls = None
		self.__t_ee = None
		self.__r_fl_fr = [None, None]
		self.__from = None
		self.__to = None
		self.__is_critical = False

	def calculate(self):
		self.__t_ls = self.to_n().times()[1] - self.__t
		self.__t_ee = self.from_n().times()[0] + self.__t
		self.__r_fl_fr[0] = self.to_n().times()[1] - self.from_n().times()[0] - self.__t
		self.__r_fl_fr[1] = self.to_n().times()[0] - self.from_n().times()[0] - self.__t

	def reset_parameters(self):
		self.__t_ls = None
		self.__t_ee = None
		self.__r_fl_fr = [None, None]

	def set_critical(self, v = True):
		if self.from_n().is_critical() and self.to_n().is_critical():
			self.__is_critical = v

	def is_critical(self):
		return self.__is_critical

	def tls_tee_rfl_rfr(self):
		return [self.__t_ls, self.__t_ee, self.__r_fl_fr[0], self.__r_fl_fr[1]]

	def to_n(self):
		return self.__to

	def from_n(self):
		return self.__from

	def set_to_n(self, n):
		self.__to = n
		return self

	def set_from_n(self, n):
		self.__from = n
		return self

	def time(self):
		return self.__t

	def name(self):
		return self.__n

	def __str__(self):
		t = self.to_n().name() if self.to_n() else 'None'
		f = self.from_n().name() if self.from_n() else 'None'
		return f'{self.time()}{self.name()}E{f}-{t}'


def find_n_with(nodes_to_search_in, edges_to_search, flag = 'ins'):
	ens = [e.name() for e in edges_to_search]
	ret_nodes = []
	for n in nodes_to_search_in:
		cur_es = [b.name() for b in n.ins()]
		if flag == 'outs':
			cur_es = [b.name() for b in n.outs()]
		if set(ens) == set(cur_es):
			ret_nodes.append(n)
	if ret_nodes:
		return ret_nodes
	return False


def find_start(ns):
	res = find_n_with(ns, [], flag = 'ins')
	if not res:
		return False
	if len(res) == 1:
		return res[0]
	return False


def find_end(ns):
	res = find_n_with(ns, [], flag = 'outs')
	if not res:
		return False
	if len(res) == 1:
		return res[0]
	return False


def find_endless_edges(ns):
	ends = [eo for eo in sum([n.outs() for n in ns], []) if not eo.to_n()]
	return ends


def find_startless_edges(ns):
	starts = [ei for ei in sum([n.ins() for n in ns], []) if not ei.from_n()]
	return starts


def link(ins, n, outs):
	n.set_ins(ins)
	n.set_outs(outs)
	return n


class Graph:
	def __init__(self):
		self.__sn = None
		self.__ns = None
		self.__cpt = None
		self.__cps = []

	def remove_node(self, node):
		self.__ns.remove(node)

	def add_cp(self, edges):
		self.__cps.append(edges)

	def update_all_id(self):
		self.__sn.update().update_id(1)
		find_end(self.nodes()).update().update_id(len(self.nodes()))
		i = 2
		for n in self.nodes():
			n.update()
			if not n.is_start() and not n.is_end():
				n.update_id(i)
				i += 1

	def already_has_cp(self, moves):
		for cp in self.cps():
			if cp == moves:
				return True
		return False

	def complete(self):
		if not find_end(self.nodes()):
			n = Node()
			n.set_ins(find_endless_edges(self.nodes()))
			self.__ns.append(n)
		if not find_start(self.nodes()):
			n = Node()
			n.set_outs(find_startless_edges(self.nodes()))
			self.__ns.append(n)
			self.__sn = n
		else:
			self.__sn = find_start(self.nodes())

	def cpt(self):
		return self.__cpt

	def cpn(self):
		return len(self.__cps)

	def cps(self):
		return self.__cps

	def nodes(self):
		return self.__ns

	def edges(self):
		pushed = []
		result = []
		for n in self.nodes():
			for e in n.ins():
				if id(e) not in pushed:
					result.append(e)
					pushed.append(id(e))
			for e in n.outs():
				if id(e) not in pushed:
					result.append(e)
					pushed.append(id(e))
		return result

	def set_nodes(self, ns):
		self.__ns = ns
		self.__sn = find_start(self.nodes())

	def merge(self):
		for n in self.nodes():
			def rule(test_n):
				return Node.can_merge_ns(n, test_n)

			for n2 in list(filter(rule, self.nodes())):
				n.merge(n2)
				self.__ns.remove(n2)

	def reset_parameters(self):
		for n in self.__ns:
			n.reset_parameters()
		for e in self.edges():
			e.reset_parameters()

	def calculate_parameters(self):
		find_end(self.nodes()).calculate_te()
		find_start(self.nodes()).calculate_tl()
		for n in self.nodes():
			n.calculate_r()
		for e in self.edges():
			e.calculate()

	def pave_critical_path(self):
		cur_node = self.__sn
		self.__cpt = 0
		while cur_node.outs():
			ne = Node.critical_path(cur_node)[1]
			ne.set_critical()
			self.__cpt += ne.time()
			cur_node = ne.to_n()

	def check_other_paths(self, aim):
		g = copy.deepcopy(self)
		nodes = g.nodes()
		for n in nodes:
			if n.times()[2] != 0:
				for ei in n.ins():
					ei.from_n().delete_out(ei)
				for eo in n.outs():
					eo.to_n().delete_in(eo)
				g.remove_node(n)
		for n in g.nodes():
			if n.times()[2] != 0:
				for ei in n.ins():
					ei.from_n().delete_out(ei)
				for eo in n.outs():
					eo.to_n().delete_in(eo)
				g.remove_node(n)
		self.merge()
		self.complete()
		self.update_all_id()
		g.reset_parameters()
		g.calculate_parameters()
		g.__sn = find_start(g.__ns)
		g.__sn.longest_path_to_end(self, aim, 0, [])

	def organize(self):
		self.merge()
		self.complete()
		self.update_all_id()
		self.calculate_parameters()
		self.pave_critical_path()
		self.check_other_paths(self.cpt())

	def graphs(self, save = False, name = ''):
		for i, cp in enumerate(self.cps()):
			show_graph_new(self, cp, i + 1, main_critical = True if i == 0 else False, save = save, name = name)
