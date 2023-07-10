from pysat.card import ITotalizer


def atLeatK_to_cnfPlus_without_new_vars(path_cnf, path_res_cnf):
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
        bound = int(info[-3])
        res_var = int(info[-1])
        if bound <= 0:
            k_clauses += 1
            file_res.write(str(res_var) + ' 0\n')
            continue
        vars = []
        for i in info:
            if i == ">=":
                break
            vars.append(int(i))
        if len(vars) < bound:
            k_clauses += 1
            file_res.write(str(-res_var) + ' 0\n')
            continue

        k_clauses += 1
        for var in vars:
            file_res.write(str(var) + ' ')
        for i in range(bound):
            file_res.write(str(-res_var) + ' ')
        file_res.write('>= ' + str(bound) + '\n')

        k_clauses += 1
        for var in vars:
            file_res.write(str(var) + ' ')
        for i in range(len(vars) - bound + 1):
            file_res.write(str(-res_var) + ' ')
        file_res.write('<= ' + str(len(vars)) + '\n')
    file.close()
    file_res.close()

    with open(path_res_cnf,
              'r+') as fp:  # Там есть ограничение на длину списка в 536 870 912, так что поаккуратнее с большими формулами
        lines = fp.readlines()
        lines.insert(0, 'p cnf+ ' + str(k_vars) + ' ' + str(k_clauses) + '\n')
        fp.seek(0)
        fp.writelines(lines)


# atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/equal_models_0_and_0.cnf", "../../data/CNFs/equal_models_0_and_0_small.cnfp")
# atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/equal_models_0_and_4.cnf", "../../data/CNFs/equal_models_0_and_4_small.cnfp")
# atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/equal_models_0_and_5.cnf", "../../data/CNFs/equal_models_0_and_5_small.cnfp")
# atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/equal_models_4_and_0.cnf", "../../data/CNFs/equal_models_4_and_0_small.cnfp")
# atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/equal_models_4_and_4.cnf", "../../data/CNFs/equal_models_4_and_4_small.cnfp")
# atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/equal_models_4_and_5.cnf", "../../data/CNFs/equal_models_4_and_5_small.cnfp")
# atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/equal_models_5_and_0.cnf", "../../data/CNFs/equal_models_5_and_0_small.cnfp")
# atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/equal_models_5_and_4.cnf", "../../data/CNFs/equal_models_5_and_4_small.cnfp")
atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/equal_models_5_and_5.cnf", "../../data/CNFs/equal_models_5_and_5_small.cnfp")

# atLeatK_to_cnfPlus_without_new_vars("../../data/CNFs/check2.cnf", "../../data/CNFs/check2.cnfp")
