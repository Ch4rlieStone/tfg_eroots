from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.operators.sampling import get_sampling
from pymoo.operators.crossover import get_crossover
from pymoo.operators.mutation import get_mutation
from pymoo.optimize import minimize
from NSGA_3.problem import MyProblem

# Reference directions for NSGA-III
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# Instantiate the algorithm
algorithm = NSGA3(
    pop_size=100,
    ref_dirs=ref_dirs,
    sampling=get_sampling("int_random"),
    crossover=get_crossover("int_sbx", prob=0.9, eta=15),
    mutation=get_mutation("int_pm", eta=20)
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
