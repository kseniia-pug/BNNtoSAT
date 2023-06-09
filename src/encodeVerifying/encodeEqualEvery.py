def make_cnf_equal_models(name1, len_input_1, id_res1, k_res1, name2, len_input_2, id_res2, k_res2):
    k_vars = 0
    k_clauses = 0
    n1 = 0

    assert len_input_1 == len_input_2
    assert k_res1 == k_res2

    file1 = open('../../data/CNFs/' + name1 + '.cnf', 'r')
    file2 = open('../../data/CNFs/' + name2 + '.cnf', 'r')
    file_res = open('../../data/CNFs/equal_' + name1 + '_and_' + name2 + '.cnf', 'w')
    while True:
        line = file1.readline()
        if not line:
            break
        if line[0] == 'p':
            info = line.split()
            k_vars = int(info[-2])
            n1 = int(info[-2])
            k_clauses = int(info[-1])
            continue
        file_res.write(line)

    while True:
        line = file2.readline()
        if not line:
            break
        if line[0] == 'c':
            continue
        info = line.split()
        if line[0] == 'p':
            k_vars += int(info[-2]) - len_input_2
            k_clauses += int(info[-1])
            continue
        for var in info:
            if var == '0' or var == '>=' or var == '<=':
                break
            if int(var) <= len_input_2:
                file_res.write(var + ' ')
            else:
                file_res.write(str(int(var) - len_input_2 + n1) + ' ')
        if info[-1] == '0':
            file_res.write('0\n')
        else:
            assert int(info[-1]) - len_input_2 + n1 > 0
            file_res.write(info[-4] + ' ' + info[-3] + ' ' + info[-2] + str(int(info[-1]) - len_input_2 + n1) + '\n')

    res_clause = []
    for i in range(k_res1):
        k_vars += 1
        k_clauses += 4
        res_clause.append(k_vars)
        file_res.write(str(id_res1 + i) + ' ' + str(id_res2 + i) + ' ' + str(-k_vars) + '\n')
        file_res.write(str(id_res1 + i) + ' ' + str(-id_res2 - i) + ' ' + str(k_vars) + '\n')
        file_res.write(str(-id_res1 - i) + ' ' + str(id_res2 + i) + ' ' + str(k_vars) + '\n')
        file_res.write(str(-id_res1 - i) + ' ' + str(-id_res2 - i) + ' ' + str(-k_vars) + '\n')

    for var in res_clause:
        file_res.write(str(var) + ' ')
    file_res.write('0\n')

    file1.close()
    file2.close()
    file_res.close()


make_cnf_equal_models(name1='check2', len_input_1=3, id_res1=4, k_res1=1, name2='check2_cnf', len_input_2=3, id_res2=4, k_res2=1)
