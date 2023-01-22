class Candidate:
	def __init__(self, n, ws):
		self.__n = n
		self.__ws = ws

	def ws(self):
		return self.__ws

	def __str__(self):
		return self.__n


def norm_list_v2(w):
	s = sum(it[1] for it in w)
	return [[it[0], it[1] / s] for it in w]
