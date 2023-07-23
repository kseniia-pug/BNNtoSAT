from pysat.solvers import Solver
from pysat.formula import CNFPlus
from pysat.formula import CNF


def solve(path, mode='cnf', solver='glucose4'):  # mode: 'cnf'|'cnfp'
    print('-----------------------')
    print(path)
    file = open(path, 'r')
    print(file.readline())
    cnf = CNF(from_file=path) if mode == 'cnf' else CNFPlus(from_file=path)
    print('End parse cnf')

    s = Solver(name=solver, bootstrap_with=cnf, use_timer=True)
    if s.solve():
        print("SAT")
    else:
        print("UNSAT")
    print(s.accum_stats())
    print('{0:.5f}s'.format(s.time()))
    s.delete()


solve("../../data/CNFs/universal_adversarial_robustness_model0.cnfp", mode='cnfp', solver='minicard')
