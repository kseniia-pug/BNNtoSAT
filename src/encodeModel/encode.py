import math
import sys


# + Init
# QuantConv2DMaxPooling2D
# QuantConv2D
# + Flatten
# + QuantDense
# + Output

class Var:
    def __init__(self, name, id):
        self.name = name
        self.id = id


class Constraint:
    def __init__(self):
        self.vars = []
        self.c = 0
        self.res = None


def print_clauses(clauses, file=sys.stdout):
    for clause in clauses:
        if len(clause) > 0:
            for c in clause:
                file.write(str(c) + ' ')
            file.write('0' + '\n')

def print_constraints(constraints, file=sys.stdout):
    for constraint in constraints:
        if len(constraint.vars) == 0:
            assert constraint.c == 0
            file.write(str(constraint.res) + ' 0\n')
        elif len(constraint.vars) < constraint.c:
            file.write(str(-constraint.res) + ' 0\n')
        else:
            for var in constraint.vars:
                file.write(str(var) + ' ')
            file.write('>= ' + str(int(constraint.c)) + ' # ' + str(constraint.res) + '\n')

class Encode:
    def __init__(self, layers, input_shape=(28, 28, 1), id_start=1):
        assert id_start > 0

        self.layers = []
        self.all_vars = []
        self.output_vars_layers = []
        self.input_shape = input_shape
        self.output_layers_shape = []
        self.id_start = id_start
        self.constraints = []
        self.clauses = []

        if len(layers) > 0:
            last = layers[0].name.split('/')[0]
            self.layers.append([])
            i = 0
            for layer in layers:
                curr = layer.name.split('/')[0]
                if curr != last:
                    last = curr
                    self.layers.append([])
                    i += 1
                self.layers[i].append(layer)

    def change_input_vars(self, id_start):
        assert id_start < self.id_start
        for i in range(len(self.constraints)):
            if abs(self.constraints[i].res) < self.id_start + self.output_vars_layers[0][1]:
                if self.constraints[i].res < 0:
                    self.constraints[i].res = -self.constraints[i].res
                    self.constraints[i].res -= self.id_start
                    self.constraints[i].res += id_start
                    self.constraints[i].res = -self.constraints[i].res
                else:
                    self.constraints[i].res -= self.id_start
                    self.constraints[i].res += id_start
            for j in range(len(self.constraints[i].vars)):
                if abs(self.constraints[i].vars[j]) < self.id_start + self.output_vars_layers[0][1]:
                    print(str(self.constraints[i].vars[j]) + '->', end='')
                    if self.constraints[i].vars[j] < 0:
                        self.constraints[i].vars[j] = -self.constraints[i].vars[j]
                        self.constraints[i].vars[j] -= self.id_start
                        self.constraints[i].vars[j] += id_start
                        self.constraints[i].vars[j] = -self.constraints[i].vars[j]
                    else:
                        self.constraints[i].vars[j] -= self.id_start
                        self.constraints[i].vars[j] += id_start
                    print(str(self.constraints[i].vars[j]))
        for i in range(len(self.clauses)):
            for j in range(len(self.clauses[i])):
                if abs(self.clauses[i][j]) < self.id_start + self.output_vars_layers[0][1]:
                    if self.clauses[i][j] < 0:
                        self.clauses[i][j] = -self.clauses[i][j]
                        self.clauses[i][j] -= self.id_start
                        self.clauses[i][j] += id_start
                        self.clauses[i][j] = -self.clauses[i][j]
                    else:
                        self.clauses[i][j] -= self.id_start
                        self.clauses[i][j] += id_start
        for i in range(len(self.output_vars_layers)):
            self.output_vars_layers[i] = (id_start + self.output_vars_layers[0][0] - 1, self.output_vars_layers[0][1])
        for i in range(self.output_vars_layers[0][1]):
            self.all_vars[i].id = i + id_start

    def encode(self):
        print("Start encode BNN")
        for layer in self.layers:
            if layer[0].name.split('/')[1] == "Init":
                self.encodeInit(layer)
            elif layer[0].name.split('/')[1] == "QuantConv2DMaxPooling2D":
                self.encodeQuantConv2DMaxPooling2D(layer)
            elif layer[0].name.split('/')[1] == "QuantConv2D":
                self.encodeQuantConv2D(layer)
            elif layer[0].name.split('/')[1] == "Flatten":
                self.encodeFlatten(layer)
            elif layer[0].name.split('/')[1] == "QuantDense":
                self.encodeQuantDense(layer)
            elif layer[0].name.split('/')[1] == "Output":
                self.encodeOutput(layer)
        print("End encode BNN")

    def create_var(self, name):
        id = self.id_start
        if len(self.all_vars) > 0:
            id = self.all_vars[-1].id + 1
        var = Var(name, id)
        self.all_vars.append(var)
        return var

    def print_vars(self, file=sys.stdout):
        for i in self.all_vars:
            file.write('c ' + str(i.name) + ' ' + str(i.id) + '\n')

    def print_constraints(self, file=sys.stdout):
        print_constraints(self.constraints, file)

    def print_clauses(self, file=sys.stdout):
        print_clauses(self.clauses, file)

    def save_cnf(self, name, mode='w'):
        file = open(name, mode)
        if mode == 'w':
            file.write('p cnf ' + str(len(self.all_vars)) + ' ' + str(len(self.clauses) + len(self.constraints)) + '\n')
        self.print_vars(file)
        self.print_constraints(file)
        self.print_clauses(file)
        file.close()

    def encodeInit(self, layer):
        k = 1
        for i in self.input_shape:
            k *= i
        for i in range(k):
            _ = self.create_var("input")
        self.output_vars_layers.append((self.id_start, k))
        self.output_layers_shape.append(self.input_shape)
        print("encodeInit")

    def encodeQuantConv2DMaxPooling2D(self, layer):
        print("encodeQuantConv2DMaxPooling2D")

    def encodeQuantConv2D(self, layer):
        print("encodeQuantConv2D")

    def encodeFlatten(self, layer):
        k = 1
        for i in self.output_layers_shape[-1]:
            k *= i
        self.output_vars_layers.append(self.output_vars_layers[-1])
        self.output_layers_shape.append((k))
        print("encodeFlatten")

    def getCForBatchNormalization(self, batchNormalization, b=None):
        # print("moving_variance", batchNormalization.moving_variance.read_value().numpy())
        # print("epsilon", batchNormalization.epsilon)
        # print("gamma", batchNormalization.gamma)
        # print("beta", batchNormalization.beta)
        # print("moving_mean", batchNormalization.moving_mean)

        if b == None:
            b = [0] * len(batchNormalization.moving_variance.read_value().numpy())
        c = [0] * len(b)
        for i in range(len(c)):
            c_i = -(batchNormalization.moving_variance.read_value().numpy()[i] + batchNormalization.epsilon) / \
                  batchNormalization.gamma.read_value().numpy()[i] * batchNormalization.beta.read_value().numpy()[i] + \
                  batchNormalization.moving_mean.read_value().numpy()[i] - b[i]
            if batchNormalization.gamma.read_value().numpy()[i] > 0:
                c[i] = math.ceil(c_i)
            else:
                c[i] = math.floor(c_i)
        return c

    def encodeQuantDense(self, layer):
        start = self.all_vars[-1].id + 1
        k = 0
        if len(self.output_vars_layers) == 0:
            raise "First layer must be Init"
        prev_var = self.output_vars_layers[-1]
        a = layer[0].weights[0].read_value().numpy().transpose()

        # print("a", a)
        c = []
        if len(layer[0].weights) > 1:
            c = self.getCForBatchNormalization(layer[1], layer[0].weights[1].read_value().numpy())
        else:
            c = self.getCForBatchNormalization(layer[1])
        for i in range(len(a)):
            k += 1
            constraint = Constraint()
            var = self.create_var(layer[0].name)
            constraint.res = var.id
            for j in range(prev_var[1]):
                if a[i][j] == 1:
                    constraint.vars.append(j + prev_var[0])
                else:
                    constraint.c += 1
                    constraint.vars.append(-(j + prev_var[0]))
            constraint.c += (c[i] + sum(a[i])) // 2
            self.constraints.append(constraint)
        self.output_vars_layers.append((start, k))
        self.output_layers_shape.append((k))
        print("encodeQuantDense")

    def encodeOutput(self, layer):
        start = self.all_vars[-1].id + 1
        if len(self.output_vars_layers) == 0:
            raise "First layer must be Init"
        prev_var = self.output_vars_layers[-1]
        a = layer[0].weights[0].read_value().numpy().transpose()

        b = [0] * len(layer[0].weights[0].read_value().numpy()[0])
        if len(layer[0].weights) > 1:
            b = layer[0].weights[1].read_value().numpy()

        for i in range(len(a)):
            self.create_var("Ans=" + str(i))

        for i in range(len(a)):
            ans_constraint = Constraint()
            ans_constraint.c = len(a)
            ans_constraint.res = self.all_vars[start + i - self.id_start].id
            for j in range(len(a)):
                constraint = Constraint()
                var = self.create_var(layer[0].name)
                constraint.res = var.id
                ans_constraint.vars.append(var.id)
                for l in range(prev_var[1]):
                    if a[i][l] == 1 and a[j][l] == -1:
                        constraint.vars.append(l + prev_var[0])
                    elif a[i][l] == -1 and a[j][l] == 1:
                        constraint.c += 1
                        constraint.vars.append(-(l + prev_var[0]))
                constraint.c += (math.ceil(b[j] - b[i]) + sum(a[i]) - sum(a[j])) // 2
                self.constraints.append(constraint)
            self.constraints.append(ans_constraint)

        self.output_vars_layers.append((start, len(a)))
        self.output_layers_shape.append((len(a)))
        print("encodeOutput")
