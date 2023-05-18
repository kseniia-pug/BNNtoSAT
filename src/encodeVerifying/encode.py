from src.encodeModel.encode import print_clauses
from src.encodeModel.encode_constraint import encode_geq_and_leq, num2bits


def add_input_constraint(path_cnf, a, b):
    file = open(path_cnf, 'r')
    res_file = open(path_cnf[:len(path_cnf)-4] + "_with_input_constraint_" + str(a) + "_and_" + str(b) + ".cnf", 'w')

    x = []
    while True:
        line = file.readline()
        if not line:
            break
        if line[0] == 'c':
            info = line.split()
            if info[1] == 'input':
                x.append(int(info[2]))
    x = list(set(x))
    n = len(x)
    clauses = encode_geq_and_leq(x, num2bits(a, n), num2bits(b, n))
    while True:
        line = file.readline()
        if not line:
            break
        if line[0] == 'p':
            info = line.split()
            res_file.write('p cnf ' + info[2] + ' ' + str(int(info[3]) + len(clauses)) + '\n')
        else:
            res_file.write(line)
    print_clauses(clauses, res_file)

    res_file.close()
    file.close()


for a in range(0, 700, 100):
    b = a + 100
    add_input_constraint("../../data/CNFs/equal_models_0_and_0.cnf", a, b)
    add_input_constraint("../../data/CNFs/equal_models_0_and_4.cnf", a, b)
    add_input_constraint("../../data/CNFs/equal_models_0_and_5.cnf", a, b)
    add_input_constraint("../../data/CNFs/equal_models_4_and_0.cnf", a, b)
    add_input_constraint("../../data/CNFs/equal_models_4_and_4.cnf", a, b)
    add_input_constraint("../../data/CNFs/equal_models_4_and_5.cnf", a, b)
    add_input_constraint("../../data/CNFs/equal_models_5_and_0.cnf", a, b)
    add_input_constraint("../../data/CNFs/equal_models_5_and_4.cnf", a, b)
    add_input_constraint("../../data/CNFs/equal_models_5_and_5.cnf", a, b)
