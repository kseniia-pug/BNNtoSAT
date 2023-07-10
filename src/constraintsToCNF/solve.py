from pysat.solvers import Solver
from pysat.formula import CNFPlus
from pysat.formula import CNF

def solve(path_cnf, is_cnf=True, solver='glucose4'):
    print('\n\n\n-----------------------')
    print(path_cnf)
    file = open(path_cnf, 'r')
    print(file.readline())
    cnf = CNF(from_file=path_cnf) if is_cnf else CNFPlus(from_file=path_cnf)
    print('End parse cnf')

    s = Solver(name=solver, bootstrap_with=cnf, use_timer=True)
    if s.solve():
        print("SAT")
    else:
        print("UNSAT")
    print(s.accum_stats())
    print('{0:.5f}s'.format(s.time()))
    s.delete()

# solve("../../data/CNFs/equal_models_0_and_4_cnf.cnf")
# solve("../../data/CNFs/equal_models_0_and_5.cnfp")
# solve("../../data/CNFs/equal_models_4_and_0.cnfp")
# solve("../../data/CNFs/equal_models_4_and_5.cnfp")
# solve("../../data/CNFs/equal_models_5_and_0.cnfp")
# solve("../../data/CNFs/equal_models_5_and_4.cnfp")

# solve("../../data/CNFs/equal_models_0_and_0.cnfp")
# solve("../../data/CNFs/equal_models_4_and_4.cnfp")
# solve("../../data/CNFs/equal_models_5_and_5.cnfp")

# solve("../../data/CNFs/equal_models_0_and_4_cnf.cnf")
# solve("../../data/CNFs/equal_models_0_and_5_cnf.cnf")
# solve("../../data/CNFs/equal_models_4_and_0_cnf.cnf")
# solve("../../data/CNFs/equal_models_4_and_5_cnf.cnf")
# solve("../../data/CNFs/equal_models_5_and_0_cnf.cnf")
# solve("../../data/CNFs/equal_models_5_and_4_cnf.cnf")

# solve("../../data/CNFs/check.cnfp")
# solve("../../data/CNFs/equal_models_0_and_4_cnf_from_cnf+.cnf", solver='minicard')
solve("../../data/CNFs/check3.cnfp", is_cnf=False, solver='minicard')
