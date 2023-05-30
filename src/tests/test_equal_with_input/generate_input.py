import random
from sys import argv

id1, id2 = int(argv[1]), int(argv[2])
path = "../../../data/CNFs/equal_models_" + str(id1) + "_and_" + str(id2) + ".cnf"

file = open(path, 'r')
m, n, all_n = 0, 0, 0
x = []
while True:
    line = file.readline()
    if not line:
        break
    if line[0] == 'p':
        info = line.split()
        all_n = int(info[2])
        m = int(info[3])
    elif line[0] == 'c':
        info = line.split()
        if info[1] == 'input':
            x.append(int(info[2]))
x = list(set(x))
n = len(x)
assert n != 0
file.close()

input = []
for i in range(n):
    input.append(random.randint(0, 1))
    print(input[-1], end=' ')

new_path = "../../../data/CNFs/equal_models_" + str(id1) + "_and_" + str(id2) + "_input.cnf"
new_file = open(new_path, 'w')
file = open(path, 'r')
new_file.write('p cnf ' + str(all_n) + ' ' + str(m + n) + '\n')
while True:
    line = file.readline()
    if not line:
        break
    if line[0] != 'p':
        new_file.write(line)
file.close()
for i in range(n):
    if input[i] == 0:
        new_file.write('-')
    new_file.write(str(i+1) + ' 0\n')
new_file.close()
