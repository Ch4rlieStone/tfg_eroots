from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
#  from pymoo.operators.crossover.pntx import TwoPointCrossover
#  from pymoo.operators.mutation.bitflip import BitflipMutation
#  from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from prova_nsga_formulacio import binhorn_own

#  problem = get_problem("bnh")
problem = binhorn_own

algorithm = NSGA2(pop_size=50)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

print(res)

plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()