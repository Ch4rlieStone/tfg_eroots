import numpy as np#
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
#  from pymoo.operators.crossover.pntx import TwoPointCrossover
#  from pymoo.operators.mutation.bitflip import BitflipMutation
#  from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


from pymoo.core.mixed import MixedVariableGA
from pymoo.core.variable import Real, Integer, Choice, Binary
from windopti import MixedVariableProblem
from windopti_constraints import MixedVariableProblem_constraints
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.constraints.as_penalty import ConstraintsAsPenalty


problem = MixedVariableProblem()

#problem = MixedVariableProblem_constraints()

algorithm = MixedVariableGA(pop_size = 200, survival=RankAndCrowding(crowding_func="pcd"))

res = minimize(problem,
               algorithm,
               termination=('n_evals', 500),
               seed=1,
               verbose=False)
"""
res = minimize(ConstraintsAsPenalty(problem,penalty=100),
               algorithm,
               termination=('n_evals', 500),
               seed=1,
               verbose=False)
"""

#print(res.F)
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
#min_row_index = np.argmin(np.min(res.F, axis=1))
#min_row = res.F[min_row_index]
#print("min row =", min_row)
#print(res.X)
#print(res.CV)
# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
plot = Scatter()
#plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

"""
#  problem = get_problem("bnh")
problem = binhorn_own

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=1,
               verbose=False)

print(res.F)

plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()
"""