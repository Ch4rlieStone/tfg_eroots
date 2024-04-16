import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.visualization.scatter import Scatter

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.core.mixed import MixedVariableGA


problem = MixedVariableGA()
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=200)
algorithm = NSGA3(pop_size=20,
                  sampling=IntegerRandomSampling(),
                  crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                  mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                  eliminate_duplicates=True,
                  ref_dirs=ref_dirs)

res = minimize(problem,
               algorithm,
               ('n_gen', 30),
               seed=1,
               verbose=True,
               save_history=True)