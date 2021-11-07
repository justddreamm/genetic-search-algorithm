import random


class Being:
	__mut_prob: None
	__dev_amp: None
	__borders: None
	__f: None

	def __init__(self):
		self.__chromes = [0, 0]

	@classmethod
	def set_mut_prob(cls, mut_prob):
		cls.__mut_prob = mut_prob

	@classmethod
	def set_dev_amp(cls, del_per):
		cls.__dev_amp = del_per

	@classmethod
	def set_borders(cls, coordinates):
		cls.__borders = coordinates

	@classmethod
	def set_fun(cls, f):
		cls.__f = f

	@staticmethod
	def custom_sort_key(b):
		return b.fit()

	@classmethod
	def show_meta(cls):
		with open('results.txt', 'w') as file:
			file.write('Settings so far\n')
			file.write(f'Deviation share amplitude: {cls.__dev_amp}\n')
			file.write(f'Mutation probability: {cls.__mut_prob}\n')
			file.write(f'Borders:\n'
			           f'x [ {cls.__borders[0][0]} ; {cls.__borders[0][1]} ]\n'
			           f'y [ {cls.__borders[1][0]} ; {cls.__borders[1][1]} ]\n')

	@staticmethod
	def generate(n):
		return [Being().rand() for i in range(n)]

	def rand(self):
		self.set_all(random.uniform(Being.__borders[0][0], Being.__borders[0][1]), random.uniform(
			Being.__borders[1][0], Being.__borders[1][1]))
		return self

	def set(self, i, val):
		self.__chromes[i] = val
		return self

	def set_all(self, x, y):
		self.set(0, x).set(1, y)
		return self

	def get(self, i):
		return self.__chromes[i]

	def mutate(self, ci):
		val = random.random() * Being.__dev_amp * (Being.__borders[ci][1] - Being.__borders[ci][0])
		if random.random() < 0.5:
			val *= -1
		if self.get(ci) + val < Being.__borders[ci][0]:
			self.set(ci, Being.__borders[ci][0])
			return self
		if self.get(ci) + val > Being.__borders[ci][1]:
			self.set(ci, Being.__borders[ci][1])
			return self
		self.set(ci, self.get(ci) + val)
		return self

	def fit(self):
		return Being.__f(self.get(0), self.get(1))

	def values(self):
		return [self.get(0), self.get(1), self.fit()]

	def crossing(self, b):
		return [
			Being().set_all(self.get(0), b.get(1)),
			Being().set_all(b.get(0), self.get(1))
		]


class Generation:
	def __init__(self, n, bs):
		bs.sort(key = Being.custom_sort_key, reverse = True)
		self.__bs = bs
		self.__n = n

	@staticmethod
	def custom_sort_key(g):
		return g.best_fit()

	def beings(self):
		return self.__bs

	def best_fit(self):
		return self.best_being().fit()

	def max_avg_fit(self):
		return [self.best_being().fit(), sum(b.fit() for b in self.beings()) / len(self.beings())]

	def n(self):
		return self.__n

	def best_values(self):
		return self.best_being().values()

	def best_being(self):
		return self.beings()[0]


def genetic(n, g_lim, prob):
	bs = Being.generate(n)
	gen = Generation(0, bs)
	g0 = gen
	gens = [gen]
	nvs = [0]
	y0vs = [gen.max_avg_fit()[0]]
	yavs = [gen.max_avg_fit()[1]]

	for N in range(1, g_lim + 1):
		progeny = bs[0].crossing(bs[1])
		progeny.extend(bs[0].crossing(bs[2]))
		bs = progeny

		for b in bs:
			if random.random() < prob:
				b.mutate(0)
			if random.random() < prob:
				b.mutate(1)

		gen = Generation(N, bs)
		g0 = max([gen, g0], key = Generation.custom_sort_key)
		if N < 10 or N % 10 == 0:
			gens.append(gen)
			nvs.append(N)
			y0vs.append(gen.max_avg_fit()[0])
			yavs.append(gen.max_avg_fit()[1])

	return [g0, gens, [nvs, yavs, y0vs]]
