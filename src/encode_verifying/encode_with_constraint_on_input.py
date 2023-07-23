# Здесь можно построить формулу с ограничением, это просто пример
from pathlib import Path

from src.encode_model.encode_model import print_clauses
from src.encode_model.encode_constraint import encode_geq_and_leq, str2bits


def add_constraint_on_input(path, a, b):
    file = open(path, 'r')
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
    clauses = encode_geq_and_leq(x, str2bits(a, n), str2bits(b, n))
    file.close()

    res_file = open("../../data/CNFs/" + Path(path).stem + "_with_constraint_on_input_" + a + "_and_" + b + Path(path).suffix, 'w')
    file = open(path, 'r')
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


a = '0'
b = '10000000000000000'
add_constraint_on_input("../../data/CNFs/universal_adversarial_robustness_model0.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/universal_adversarial_robustness_model4.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/universal_adversarial_robustness_model5.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/equality_model0_and_model0.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/equality_model0_and_model4.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/equality_model0_and_model5.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/equality_model4_and_model0.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/equality_model4_and_model4.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/equality_model4_and_model5.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/equality_model5_and_model0.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/equality_model5_and_model4.cnfcc", a, b)
add_constraint_on_input("../../data/CNFs/equality_model5_and_model5.cnfcc", a, b)
