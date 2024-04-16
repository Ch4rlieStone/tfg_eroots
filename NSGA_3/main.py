from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from problem import MyProblem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair

# Reference directions for NSGA-III
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# Instantiate the algorithm
algorithm = NSGA3(
    pop_size=100,
    ref_dirs=ref_dirs,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    
)

# Define termination criteria
termination = get_termination("n_gen", 40)

# Optimization
res = minimize(
    MyProblem(),
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)

# Print the results
print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
