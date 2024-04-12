import numpy as np
from pymoo.problems.functional import FunctionalProblem

objs = [
    lambda x: 4 * x[0] ** 2 + 4 * x[1] ** 2,
    lambda x: (x[0] - 5) ** 2 + (x[1] - 5) ** 2
]

constr_ieq = [
    lambda x: (1 / 25) * ((x[0] - 5) ** 2 + x[1] ** 2 - 25),
    lambda x: (-1 / 7.7) * ((x[0] - 8) ** 2 + (x[1] + 3) ** 2 - 7.7)
]

n_var = 2

binhorn_own = FunctionalProblem(n_var,
                            objs,
                            constr_ieq=constr_ieq,
                            xl=np.array([0, 0]),
                            xu=np.array([5, 3])
                            )


F, G = binhorn_own.evaluate(np.random.rand(3,2))

#print(f"F: {F}\n")
#print(f"G: {G}\n")