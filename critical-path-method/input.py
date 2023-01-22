from cpm import *

B = {
	'a': Edge('a', 3),
	'b': Edge('b', 5),
	'c': Edge('c', 2),
	'd': Edge('d', 4),
	'e': Edge('e', 3),
	'f': Edge('f', 1),
	'g': Edge('g', 4),
	'h': Edge('h', 3),
	'i': Edge('i', 3),
	'j': Edge('j', 2),
	'k': Edge('k', 5)
}

N = [
	link([], Node(), [B['a']]),
	link([], Node(), [B['b']]),
	link([B['a']], Node(), [B['c']]),
	link([], Node(), [B['d']]),
	link([B['b'], B['c']], Node(), [B['e']]),
	link([B['a']], Node(), [B['f']]),
	link([B['b'], B['c']], Node(), [B['g']]),
	link([B['a']], Node(), [B['h']]),
	link([B['h'], B['g']], Node(), [B['i']]),
	link([B['f'], B['e'], B['d']], Node(), [B['j']]),
	link([B['b'], B['c']], Node(), [B['k']])
]

# Nikita
# N = [
# 	link([], Node(), [B['a']]),
# 	link([], Node(), [B['b']]),
# 	link([], Node(), [B['c']]),
# 	link([B['a']], Node(), [B['d']]),
# 	link([B['b']], Node(), [B['e']]),
# 	link([B['b']], Node(), [B['f']]),
# 	link([B['d']], Node(), [B['g']]),
# 	link([B['e']], Node(), [B['h']]),
# 	link([B['f'], B['c']], Node(), [B['i']]),
# 	link([B['g']], Node(), [B['j']]),
# 	link([B['h'], B['i']], Node(), [B['k']])
# ]

# Roma
# N = [
# 	link([], Node(), [B['a']]),
# 	link([B['a']], Node(), [B['b']]),
# 	link([B['b'], B['f']], Node(), [B['c']]),
# 	link([B['c'], B['h']], Node(), [B['d']]),
# 	link([], Node(), [B['e']]),
# 	link([B['e']], Node(), [B['f']]),
# 	link([B['e']], Node(), [B['g']]),
# 	link([B['g'], B['k']], Node(), [B['h']]),
# 	link([], Node(), [B['i']]),
# 	link([B['g'], B['k']], Node(), [B['j']]),
# 	link([B['i']], Node(), [B['k']])
# ]

# Olya
# N = [
# 	link([], Node(), [B['a']]),
# 	link([], Node(), [B['b']]),
# 	link([B['a']], Node(), [B['c']]),
# 	link([B['c']], Node(), [B['d']]),
# 	link([B['a']], Node(), [B['e']]),
# 	link([B['e'], B['d']], Node(), [B['f']]),
# 	link([B['c']], Node(), [B['g']]),
# 	link([B['g']], Node(), [B['h']]),
# 	link([B['h'], B['j'], B['b'], B['f']], Node(), [B['i']]),
# 	link([B['c']], Node(), [B['j']]),
# 	link([B['g']], Node(), [B['k']])
# ]

# Vanya
# N = [
# 	link([], Node(), [B['a']]),
# 	link([], Node(), [B['b']]),
# 	link([], Node(), [B['c']]),
# 	link([B['c']], Node(), [B['d']]),
# 	link([B['c']], Node(), [B['e']]),
# 	link([B['a']], Node(), [B['f']]),
# 	link([B['a']], Node(), [B['g']]),
# 	link([B['d'], B['g']], Node(), [B['h']]),
# 	link([B['h'], B['e']], Node(), [B['i']]),
# 	link([B['b'], B['f'], B['i']], Node(), [B['j']]),
# 	link([B['e'], B['h']], Node(), [B['k']])
# ]

g = Graph()
g.set_nodes(N)
