from pysat.solvers import Solver
from pysat.formula import CNFPlus
from pysat.formula import CNF

def solve(path_cnf):
    print('\n\n\n-----------------------')
    print(path_cnf)
    file = open(path_cnf, 'r')
    print(file.readline())

    cnf = CNFPlus(from_file=path_cnf)
    # cnf = CNF(from_file=path_cnf)
    print('End parse cnf')

    s = Solver(name='minicard', bootstrap_with=cnf, use_timer=True)
    print(s.solve())
    print(s.accum_stats())
    print('{0:.5f}s'.format(s.time()))
    s.delete()

solve("../../data/CNFs/equal_models_0_and_4.cnfp")
solve("../../data/CNFs/equal_models_0_and_5.cnfp")
solve("../../data/CNFs/equal_models_4_and_0.cnfp")
solve("../../data/CNFs/equal_models_4_and_5.cnfp")
solve("../../data/CNFs/equal_models_5_and_0.cnfp")
solve("../../data/CNFs/equal_models_5_and_4.cnfp")

# solve("../../data/CNFs/equal_models_0_and_0.cnfp")
# solve("../../data/CNFs/equal_models_4_and_4.cnfp")
# solve("../../data/CNFs/equal_models_5_and_5.cnfp")

# solve("../../data/CNFs/equal_models_0_and_4_cnf.cnf")
# solve("../../data/CNFs/equal_models_0_and_5_cnf.cnf")
# solve("../../data/CNFs/equal_models_4_and_0_cnf.cnf")
# solve("../../data/CNFs/equal_models_4_and_5_cnf.cnf")
# solve("../../data/CNFs/equal_models_5_and_0_cnf.cnf")
# solve("../../data/CNFs/equal_models_5_and_4_cnf.cnf")

solve("../../data/CNFs/equal_models_0_and_4_small.cnfp")
solve("../../data/CNFs/equal_models_0_and_5_small.cnfp")
solve("../../data/CNFs/equal_models_4_and_0_small.cnfp")
solve("../../data/CNFs/equal_models_4_and_5_small.cnfp")
solve("../../data/CNFs/equal_models_5_and_0_small.cnfp")
solve("../../data/CNFs/equal_models_5_and_4_small.cnfp")

# solve("../../data/CNFs/equal_models_0_and_0_small.cnfp")
# solve("../../data/CNFs/equal_models_4_and_4_small.cnfp")
# solve("../../data/CNFs/equal_models_5_and_5_small.cnfp")

# solve("../../data/CNFs/check.cnfp")
