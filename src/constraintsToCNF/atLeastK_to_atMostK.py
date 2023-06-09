def atLeastK_to_atMostK(path_cnf, path_res_cnf):
    file = open(path_cnf, 'r')
    file_res = open(path_res_cnf, 'w')
    while True:
        line = file.readline()
        if not line:
            break

        if line[0] == 'c' or line[0] == 'p':
            file_res.write(line)
            continue

        info = line.split()
        if info[-2] != '#':
            file_res.write(line)
            continue

        n = len(info) - 4
        if n - int(info[-3]) < 0:
            file_res.write(str(-int(info[-1])) + ' 0\n')
            continue

        for i in info:
            if i == ">=":
                break
            file_res.write(str(-int(i)) + ' ')
        file_res.write('<= ' + str(n - int(info[-3])) + ' # ' + info[-1] + '\n')
    file.close()
    file_res.close()


# atLeastK_to_atMostK("../../data/CNFs/equal_models_0_and_0.cnf", "../../data/CNFs/equal_models_0_and_0_atMostK.cnf")
# atLeastK_to_atMostK("../../data/CNFs/equal_models_0_and_4.cnf", "../../data/CNFs/equal_models_0_and_4_atMostK.cnf")
# atLeastK_to_atMostK("../../data/CNFs/equal_models_0_and_5.cnf", "../../data/CNFs/equal_models_0_and_5_atMostK.cnf")
# atLeastK_to_atMostK("../../data/CNFs/equal_models_4_and_0.cnf", "../../data/CNFs/equal_models_4_and_0_atMostK.cnf")
# atLeastK_to_atMostK("../../data/CNFs/equal_models_4_and_4.cnf", "../../data/CNFs/equal_models_4_and_4_atMostK.cnf")
# atLeastK_to_atMostK("../../data/CNFs/equal_models_4_and_5.cnf", "../../data/CNFs/equal_models_4_and_5_atMostK.cnf")
# atLeastK_to_atMostK("../../data/CNFs/equal_models_5_and_0.cnf", "../../data/CNFs/equal_models_5_and_0_atMostK.cnf")
# atLeastK_to_atMostK("../../data/CNFs/equal_models_5_and_4.cnf", "../../data/CNFs/equal_models_5_and_4_atMostK.cnf")
# atLeastK_to_atMostK("../../data/CNFs/equal_models_5_and_5.cnf", "../../data/CNFs/equal_models_5_and_5_atMostK.cnf")
