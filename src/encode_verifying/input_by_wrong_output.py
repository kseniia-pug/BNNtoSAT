# вход, на котором сеть распознает не верно
# выход не равен тому, что должно. Вход отклоняется не более, чем на rho1, и не менее, чем на rho2
import subprocess

import numpy as np
import tensorflow as tf
from larq.quantizers import SteSign
from pathlib import Path

from src.encode_model.encode_model import Encode, Constraint

ZERO = [[[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],

           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],

           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1]],

           [[-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1]],

           [[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],

           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1], [1], [1], [1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],

           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],
           [[-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]],
           ]


def print_img(s):
    new_pixels = []
    for i in range(len(s)):
        for j in range(len(s[0])):
            if s[i][j][0] < 0:
                new_pixels.append('█')
            else:
                new_pixels.append(' ')

    for i in range(len(new_pixels)):
        if i % len(s) == 0:
            print('\n', end='')
        print(new_pixels[i], end='')


def print_res(path, reference_input, output, rho1, rho2=0, is_strongly=False):
    if is_strongly:
        res_vars = input_by_output_strongly(path, reference_input, output, rho1, rho2)
    else:
        res_vars = input_by_output(path, reference_input, output, rho1, rho2)

    subprocess.run(["./../../solvers/minisatcs/minisat", "../../data/CNFs/input_by_output_" + Path(path).stem + ".cnfcc", "../../solvers/res/res_input_by_output.txt"])
    file = open("../../solvers/res/res_input_by_output.txt", 'r')
    if not file.readline():
        return
    res = file.readline()
    if not res:
        return

    pixels = res.split()
    pixels = list(map(int, pixels))

    print([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(list(map(lambda num: int(num > 0), pixels[res_vars[0]-1:res_vars[0]-1 + res_vars[1]])))
    print(list(map(lambda num: abs(int(num)), pixels[res_vars[0] - 1:res_vars[0] - 1 + res_vars[1]])))

    pixels = pixels[:28 * 28]

    new_pixels = []
    for pixel in pixels:
        if pixel < 0:
            new_pixels.append('█')
        else:
            new_pixels.append(' ')

    for i in range(len(new_pixels)):
        if i % 28 == 0:
            print('\n', end='')
        print(new_pixels[i], end='')


def input_by_output_strongly(path, reference_input, output, rho1, rho2=0):
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})
    encode = Encode(model)

    assert encode.output_layers_shape[-1] == output.shape
    assert encode.input_shape == reference_input.shape

    reference_input = reference_input.flatten()
    output = output.flatten()

    for i in range(len(output)):
        if output[i] == 1:
            encode.clauses.append([-(encode.output_vars_layers[-1][0] + i)])
        else:
            encode.clauses.append([encode.output_vars_layers[-1][0] + i])

    constraint = Constraint()
    res = encode.create_var('variance').id
    encode.clauses.append([res])
    constraint.res = res
    constraint.c = int(len(reference_input) * (1 - rho1))
    for i in range(len(reference_input)):
        if reference_input[i] == 1:
            constraint.vars.append(encode.output_vars_layers[0][0] + i)
        else:
            constraint.vars.append(-(encode.output_vars_layers[0][0] + i))
    encode.constraints.append(constraint)

    constraint = Constraint()
    res = encode.create_var('variance').id
    encode.clauses.append([res])
    constraint.res = res
    constraint.c = int(len(reference_input) * rho2)
    for i in range(len(reference_input)):
        if reference_input[i] == 1:
            constraint.vars.append(-(encode.output_vars_layers[0][0] + i))
        else:
            constraint.vars.append(encode.output_vars_layers[0][0] + i)
    encode.constraints.append(constraint)

    res_path = "../../data/CNFs/input_by_output_" + Path(path).stem + ".cnfcc"
    encode.save(res_path)
    return encode.output_vars_layers[-1]


def input_by_output(path, reference_input, output, rho1, rho2=0):
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})
    encode = Encode(model)

    assert encode.output_layers_shape[-1] == output.shape
    assert encode.input_shape == reference_input.shape

    reference_input = reference_input.flatten()
    output = output.flatten()

    for i in range(len(output)):
        if output[i] == 1:
            encode.clauses.append([-(encode.output_vars_layers[-1][0] + i)])

    constraint = Constraint()
    res = encode.create_var('variance').id
    encode.clauses.append([res])
    constraint.res = res
    constraint.c = int(len(reference_input) * (1 - rho1))
    for i in range(len(reference_input)):
        if reference_input[i] == 1:
            constraint.vars.append(encode.output_vars_layers[0][0] + i)
        else:
            constraint.vars.append(-(encode.output_vars_layers[0][0] + i))
    encode.constraints.append(constraint)

    constraint = Constraint()
    res = encode.create_var('variance').id
    encode.clauses.append([res])
    constraint.res = res
    constraint.c = int(len(reference_input) * rho2)
    for i in range(len(reference_input)):
        if reference_input[i] == 1:
            constraint.vars.append(-(encode.output_vars_layers[0][0] + i))
        else:
            constraint.vars.append(encode.output_vars_layers[0][0] + i)
    encode.constraints.append(constraint)

    res_path = "../../data/CNFs/input_by_output_" + Path(path).stem + ".cnfcc"
    encode.save(res_path)
    return encode.output_vars_layers[-1]


# input_by_output("../../data/models/model0.h5", np.array(ZERO), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0.2)
# print_res("../../data/models/model0.h5", np.array(ZERO), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0.1, is_strongly=True)
print_res("../../data/models/model6.h5", np.array(ZERO), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0.2)
