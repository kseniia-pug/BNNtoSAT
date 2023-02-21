import numpy as np
import math
from multipledispatch import dispatch


class Layer(object):
    def __init__(self, a, b, alpha, sigma, mu, gamma):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.sigma = sigma
        self.mu = mu
        self.gamma = gamma
        self.sizeIn = np.shape(a)[1]
        self.sizeOut = np.shape(a)[0]
        self.input = []
        self.output = []


class CNF(object):
    def __init__(self, variables=[], constraints=[]):
        self.variables = variables
        self.constraints = constraints


class BNN(object):
    def __init__(self, layers, o, cnf=CNF([], [])):
        self.size = len(layers)
        self.layers = layers
        self.cnf = cnf
        self.o = o


def createBNNtoCNF(layers):
    v = 1
    variables = []
    constraints = []
    for _ in range(layers[0].sizeIn):
        variables += [v]
        layers[0].input += [v]
        v += 1
    for k in range(len(layers) - 1):
        for _ in range(layers[k].sizeOut):
            variables += [v]
            layers[k].output += [v]
            layers[k + 1].input += [v]
            v += 1

    for k in range(len(layers) - 1):
        l = layers[k]
        for i in range(l.sizeOut):
            constraint = []
            for j in range(l.sizeIn):
                if l.a[i][j] != 0:
                    constraint += [l.input[j]]
            c = -l.sigma[i] / l.alpha[i] * l.gamma[i] + l.mu[i] - l.b[i]
            if l.alpha[i] > 0:
                c = math.ceil(c)
            else:
                c = math.floor(c)
            constraint += ['>=', c, '#', layers[k].output[i]]
            constraints += [constraint]

    l = layers[-1]
    D = []
    for _ in range(l.sizeOut):
        d = []
        for _ in range(l.sizeOut):
            variables += [v]
            d += [v]
            v += 1
        D += [d]
    sizeO = len(bin(l.sizeOut)[:2])
    o = []
    for _ in range(sizeO):
        variables += [v]
        o += [v]
        v += 1

    for i in range(l.sizeOut):
        for j in range(l.sizeOut):
            constraint = []
            for k in range(l.sizeIn):
                x = l.a[i][k] - l.a[j][k]
                if x == 1:
                    constraint += [l.input[k]]
                elif x == -1:
                    constraint += [-l.input[k]]
            if constraint:
                constraint += ['>=', math.ceil(l.b[j] - l.b[i]), '#', D[i][j]]
                constraints += [constraint]
    for i in range(l.sizeOut):
        constraint = []
        for j in range(l.sizeOut):
            constraint += [D[i][j]]
        res = bin(i)[2:]
        constraintEnd = []
        for k in range(len(res)):
            if res[k] == '1':
                constraintEnd += [['>=', l.sizeOut, '#', o[k]]]
            else:
                constraintEnd += [['>=', l.sizeOut, '#', -o[k]]]
        for k in range(len(res), sizeO):
            constraintEnd += [['>=', l.sizeOut, '#', -o[k]]]
        for k in range(np.shape(constraintEnd)[0]):
            constraints += [constraint + constraintEnd[k]]
    return BNN(layers, o, CNF(variables, constraints))


def createEquivalence(bnn1, bnn2):
    if len(bnn1.o) != len(bnn2.o) or bnn1.layers[0].sizeIn != bnn2.layers[0].sizeIn:
        return None
    if bnn1.cnf.variables[-1] > bnn2.cnf.variables[-1]:
        bnn1, bnn2 = bnn2, bnn1
    add = bnn1.cnf.variables[-1] - bnn2.layers[0].sizeIn
    for i in range(bnn2.layers[0].sizeIn, len(bnn2.cnf.variables)):
        bnn2.cnf.variables[i] += add
    for i in range(len(bnn2.cnf.constraints)):
        for j in range(len(bnn2.cnf.constraints[i])):
            if bnn2.cnf.constraints[i][j] != '>=' and bnn2.cnf.constraints[i][j] != '#' and (j == 0 or bnn2.cnf.constraints[i][j-1] != '>=') and abs(bnn2.cnf.constraints[i][j]) > bnn2.layers[0].input[-1]:
                if bnn2.cnf.constraints[i][j] < 0:
                    bnn2.cnf.constraints[i][j] -= add
                else:
                    bnn2.cnf.constraints[i][j] += add

    constraints = bnn1.cnf.constraints + bnn2.cnf.constraints
    for i in range(len(bnn1.o)):
        constraints += [[bnn1.o[i], bnn2.o[i]]]
        constraints += [[-bnn1.o[i], -bnn2.o[i]]]
    return CNF(bnn1.cnf.variables + bnn2.cnf.variables[bnn2.layers[0].sizeIn:], constraints)


@dispatch(str, BNN)
def writeCNF(file, bnn):
    writeCNF(file, bnn.cnf)


@dispatch(str, CNF)
def writeCNF(file, cnf):
    out = open(file, 'w')
    out.write('p cnf ' + str(len(cnf.variables)) + ' ' + str(len(cnf.constraints)) + '\n')
    for i in range(len(cnf.constraints)):
        constraint = ''
        for j in range(len(cnf.constraints[i])):
            constraint += str(cnf.constraints[i][j]) + ' '
        if cnf.constraints[i][-2] != '#':
            constraint += '0'
        else:
            constraint = constraint[:-1]
        out.write(constraint + '\n')
    out.close()


a = [[[1, 1, 0, 0, 1], [1, 0, 1, 1, 1], [0, 1, 0, 0, 1]],
     [[1, 1, 0], [1, 0, 1]]]
b = [[7.4, 0.2, -0.7], [-0.1, 0.8]]
alpha = [[0.1, 0.1, -0.7], []]
mu = [[0.6, -0.1, -0.2], []]
sigma = [[-0.9, 0.2, -0.4], []]
gamma = [[0.1, 0.3, -0.6], []]
layers1 = []
layers2 = []
for i in range(len(gamma)):
    layers1 += [Layer(a[i], b[i], alpha[i], sigma[i], mu[i], gamma[i])]
    layers2 += [Layer(a[i], b[i], alpha[i], sigma[i], mu[i], gamma[i])]

bnn1 = createBNNtoCNF(layers1)
writeCNF('last1.cfn', bnn1)

bnn2 = createBNNtoCNF(layers2)
writeCNF('last2.cfn', bnn2)

cnf = createEquivalence(bnn1, bnn2)
writeCNF('last.cfn', cnf)
