import copy

import tensorflow as tf
from larq.quantizers import SteSign

from src.encodeModel.encode import Encode, Constraint, print_clauses, print_constraints


def encode(id, tau, rho): # tau: [|input|], rho: [0, 1]
    model = tf.keras.models.load_model("../../data/models/model" + str(id) + ".h5",
                                        custom_objects={"custom_activation": SteSign})
    encode = Encode(model.layers)
    encode.encode()
    encode_with_tau = Encode(model.layers, id_start=encode.all_vars[-1].id + 1)
    encode_with_tau.encode()

    input = encode_with_tau.output_vars_layers[0]
    for i in range(len(encode_with_tau.clauses)):
        for j in range(len(encode_with_tau.clauses[i])):
            if abs(encode_with_tau.clauses[i][j]) < input[1] and tau[abs(encode_with_tau.clauses[i][j])-input[0]] == 1:
                encode_with_tau.clauses[i][j] = -encode_with_tau.clauses[i][j]
    for i in range(len(encode_with_tau.constraints)):
        if abs(encode_with_tau.constraints[i].res) < input[1] and tau[abs(encode_with_tau.constraints[i].res)-input[0]] == 1:
            encode_with_tau.constraints[i].res = -encode_with_tau.constraints[i].res
        for j in range(len(encode_with_tau.constraints[i].vars)):
            if abs(encode_with_tau.constraints[i].vars[j]) < input[1] and tau[abs(encode_with_tau.constraints[i].vars[j])-input[0]] == 1:
                encode_with_tau.constraints[i].vars[j] = -encode_with_tau.constraints[i].vars[j]

    output = encode.output_vars_layers[-1]
    output_with_tau = encode_with_tau.output_vars_layers[-1]

    constraints = []
    clauses = []
    ress = []

    for k in range(output[1]):
        l = [int(x==k) for x in range(output[1])]

        constraint_res = Constraint()
        constraint_res.c = output[1]
        for i in range(output[1]):
            var = encode_with_tau.create_var('equal').id
            constraint_res.vars.append(var)
            if l[i] == 0:
                clauses.append([var, output[0] + i])
                clauses.append([-var, -(output[0] + i)])
            else:
                clauses.append([var, -(output[0] + i)])
                clauses.append([-var, output[0] + i])
        constraint_res.res = encode_with_tau.create_var('res_equal').id

        constraint = Constraint()
        constraint.c = 1
        for i in range(output[1]):
            var = encode_with_tau.create_var('not_equal').id
            constraint.vars.append(var)
            clauses.append([output[0] + i, output_with_tau[0] + i, -var])
            clauses.append([output[0] + i, -(output_with_tau[0] + i), var])
            clauses.append([-(output[0] + i), output_with_tau[0] + i, var])
            clauses.append([-(output[0] + i), -(output_with_tau[0] + i), -var])
        constraint.res = encode_with_tau.create_var('res_not_equal').id

        var = encode_with_tau.create_var('partial_res').id
        clauses.append([constraint_res.res, constraint.res, -var])
        clauses.append([constraint_res.res, -constraint.res, -var])
        clauses.append([-constraint_res.res, constraint.res, -var])
        clauses.append([-constraint_res.res, -constraint.res, var])
        ress.append(var)

        constraints.append(constraint)
        constraints.append(constraint_res)

    constraint = Constraint()
    constraint.c = int(output[1] * rho)
    constraint.res = encode_with_tau.create_var('res').id
    for i in ress:
        constraint.vars.append(i)
    clauses.append([constraint.res])
    constraints.append(constraint)

    path = "../../data/CNFs/universal_adversarial_robustness_model_" + str(id) + ".cnf"
    file = open(path, 'w')
    file.write(
        'p cnf ' + str(len(encode.all_vars) + len(encode_with_tau.all_vars)) + ' ' + str(
            len(encode.clauses) + len(encode_with_tau.constraints) + len(encode.clauses) + len(encode_with_tau.constraints) + len(
                clauses) + len(constraints)) + '\n')
    print_clauses(clauses, file)
    print_constraints(constraints, file)
    file.close()

    encode.save_cnf(path, mode='a')
    encode.save_cnf(path, mode='a')


encode(0, [int(x % 100 == 0) for x in range(784)], 0.1)
encode(4, [int(x % 1 == 0) for x in range(784)], 0.6)
encode(5, [int(x % 30 == 0) for x in range(784)], 0.4)
