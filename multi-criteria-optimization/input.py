from candidate import Candidate, norm_list_v2

W = norm_list_v2([
	['education', 8],
	['physical-condition', 2],
	['appearance', 6],
	['character', 7]
])

CANDIDATES = [
	Candidate('A. Anatoliy', [5, 5, 9, 5]),
	Candidate('B. Alexander', [7, 7, 6, 9]),
	Candidate('C. Vladimir', [4, 8, 9, 8]),
	Candidate('D. Sergey', [9, 7, 6, 7])
]
