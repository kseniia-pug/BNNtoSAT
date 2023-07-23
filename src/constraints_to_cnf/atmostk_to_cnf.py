from pysat.card import ITotalizer
from pathlib import Path


def atmostk_to_cnf(path):
    k_vars, k_clauses = 0, 0

    assert Path(path).suffix == ".cnfcc"
    path_res = path[:-len(".cnfcc")] + ".cnf"
    file = open(path, 'r')
    file_res = open(path_res, 'w')

    while True:
        line = file.readline()
        if not line:
            break

        if line[0] == 'c':
            file_res.write(line)
            continue

        info = line.split()
        if line[0] == 'p':
            k_vars = int(info[-2])
            k_clauses = int(info[-1])
            continue

        if info[-2] != '#':
            file_res.write(line)
            continue

        k_clauses -= 1
        vars = []
        for i in info:
            if i == "<=":
                break
            vars.append(int(i))
        t = ITotalizer(lits=vars, ubound=int(info[-3]), top_id=k_vars)  # длина t.rhs не обязана совпадать с ubound + 1
        for clause in t.cnf.clauses:
            k_clauses += 1
            for var in clause:
                k_vars = max(k_vars, var)
                file_res.write(str(var) + ' ')
            file_res.write('0\n')
        k_clauses += 2
        file_res.write(info[-1] + ' ' + str(t.rhs[-1]) + ' 0\n')
        file_res.write(str(-int(info[-1])) + ' ' + str(-t.rhs[-1]) + ' 0\n')
        # print(t.rhs)
        t.delete()

    file.close()
    file_res.close()

    with open(path_res, 'r+') as fp:  # Там есть ограничение на длину списка в 536 870 912, так что поаккуратнее с большими формулами
        lines = fp.readlines()
        lines.insert(0, 'p cnf ' + str(k_vars) + ' ' + str(k_clauses) + '\n')
        fp.seek(0)
        fp.writelines(lines)


atmostk_to_cnf("../../data/CNFs/universal_adversarial_robustness_model0_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/universal_adversarial_robustness_model4_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/universal_adversarial_robustness_model5_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/equality_model0_and_model0_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/equality_model0_and_model4_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/equality_model0_and_model5_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/equality_model4_and_model0_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/equality_model4_and_model4_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/equality_model4_and_model5_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/equality_model5_and_model0_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/equality_model5_and_model4_atmostk.cnfcc")
atmostk_to_cnf("../../data/CNFs/equality_model5_and_model5_atmostk.cnfcc")
