from sys import argv

file = open(argv[1], 'r')
input = []
while True:
    line = file.readline()
    if not line:
        break
    info = line.split()
    for i in info:
        input.append(int(i))
file.close()

file = open(argv[1], 'r')
res_input = []
while True:
    line = file.readline()
    if not line:
        break
    if line[0] == 'r':
        continue
    info = line.split()
    for i in info:
        if int(i) > 0:
            res_input.append(1)
        else:
            res_input.append(0)
file.close()

for i in range(len(input)):
    assert input[i] == res_input[i]
print("TEST PASS")
