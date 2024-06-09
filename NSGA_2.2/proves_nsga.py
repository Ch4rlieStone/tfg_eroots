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
from windopti_withcstr import MixedVariableProblem2
from windopti_constraints import MixedVariableProblem_constraints
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.decomposition.asf import ASF
import matplotlib.pyplot as plt
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
import time

start_time = time.time()
problem = MixedVariableProblem()
#problem = MixedVariableProblem2()


#problem = MixedVariableProblem_constraints()

algorithm = MixedVariableGA(pop_size = 300, survival=RankAndCrowding(crowding_func="cd"))

res = minimize(problem,
               algorithm,
               termination=('n_evals', 500),
               seed=1,
               verbose=False,
               save_history=True)
"""
res = minimize(ConstraintsAsPenalty(problem, penalty=100),
               algorithm,
               termination=('n_evals', 500),
               seed=1,
               verbose=False)

res = Evaluator().eval(problem, Individual(X=res.X))
"""
"""
# Plot the convergence using the history object
n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt = np.array([e.opt[0].F for e in res.history])
plt.title("Convergence")
plt.plot(n_evals, opt, "--")
plt.yscale("log")
plt.show()
"""

# Choice of decicision point (we need weights for each objective)
weights = np.array([0.5, 0.5])
decomp = ASF()
I = decomp(res.F, weights).argmin()

print(res.F)
print("Best solution found: \nX = %s\nF = %s\nC = %s" % (res.X, res.F, res.CV))
#print(res.history)

print(res.H)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time: ", execution_time,"s")
#min_row_index = np.argmin(np.min(res.F, axis=1))
#min_row = res.F[min_row_index]
#print("min row =", min_row)
#print(res.X)
#print(res.CV)
# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
plot = Scatter()


#plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="black")
# plot best point with weights aproach
plot.add(res.F[I], color="red", s=50)
print("Best solution found weigthed: \nX = %s\nF = %s" % (res.X[I], res.F[I]))
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