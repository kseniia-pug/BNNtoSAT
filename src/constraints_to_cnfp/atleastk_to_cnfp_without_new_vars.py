from pathlib import Path


def atleastk_to_cnfp_without_new_vars(path):
    k_vars, k_clauses = 0, 0

    assert Path(path).suffix == ".cnfcc"
    path_res = path[:-len(".cnfcc")] + "_without_new_vars.cnfp"
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

    with open(path_res, 'r+') as fp:  # Там есть ограничение на длину списка в 536 870 912, так что поаккуратнее с большими формулами
        lines = fp.readlines()
        lines.insert(0, 'p cnf+ ' + str(k_vars) + ' ' + str(k_clauses) + '\n')
        fp.seek(0)
        fp.writelines(lines)


atleastk_to_cnfp_without_new_vars("../../data/CNFs/universal_adversarial_robustness_model0.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/universal_adversarial_robustness_model4.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/universal_adversarial_robustness_model5.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/equality_model0_and_model0.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/equality_model0_and_model4.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/equality_model0_and_model5.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/equality_model4_and_model0.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/equality_model4_and_model4.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/equality_model4_and_model5.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/equality_model5_and_model0.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/equality_model5_and_model4.cnfcc")
atleastk_to_cnfp_without_new_vars("../../data/CNFs/equality_model5_and_model5.cnfcc")
