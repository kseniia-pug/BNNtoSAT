from pysat.card import ITotalizer


def atMostK_to_cnf(path_cnf, path_res_cnf):
    k_vars, k_clauses = 0, 0
    file = open(path_cnf, 'r')
    file_res = open(path_res_cnf, 'w')

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
        print(t.rhs)
        t.delete()

    file.close()
    file_res.close()

    with open(path_res_cnf,
              'r+') as fp:  # Там есть ограничение на длину списка в 536 870 912, так что поаккуратнее с большими формулами
        lines = fp.readlines()
        lines.insert(0, 'p cnf ' + str(k_vars) + ' ' + str(k_clauses) + '\n')
        fp.seek(0)
        fp.writelines(lines)


# atMostK_to_cnf("../../data/CNFs/equal_models_0_and_0_atMostK.cnf", "../../data/CNFs/equal_models_0_and_0_cnf.cnf")
# atMostK_to_cnf("../../data/CNFs/equal_models_0_and_4_atMostK.cnf", "../../data/CNFs/equal_models_0_and_4_cnf.cnf")
# atMostK_to_cnf("../../data/CNFs/equal_models_0_and_5_atMostK.cnf", "../../data/CNFs/equal_models_0_and_5_cnf.cnf")
# atMostK_to_cnf("../../data/CNFs/equal_models_4_and_0_atMostK.cnf", "../../data/CNFs/equal_models_4_and_0_cnf.cnf")
# atMostK_to_cnf("../../data/CNFs/equal_models_4_and_4_atMostK.cnf", "../../data/CNFs/equal_models_4_and_4_cnf.cnf")
# atMostK_to_cnf("../../data/CNFs/equal_models_4_and_5_atMostK.cnf", "../../data/CNFs/equal_models_4_and_5_cnf.cnf")
# atMostK_to_cnf("../../data/CNFs/equal_models_5_and_0_atMostK.cnf", "../../data/CNFs/equal_models_5_and_0_cnf.cnf")
# atMostK_to_cnf("../../data/CNFs/equal_models_5_and_4_atMostK.cnf", "../../data/CNFs/equal_models_5_and_4_cnf.cnf")
# atMostK_to_cnf("../../data/CNFs/equal_models_5_and_5_atMostK.cnf", "../../data/CNFs/equal_models_5_and_5_cnf.cnf")
atMostK_to_cnf("../../data/CNFs/check2.cnf", "../../data/CNFs/check2_cnf.cnf")
