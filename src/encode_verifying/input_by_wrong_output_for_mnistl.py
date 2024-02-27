# вход, на котором сеть распознает не верно
# выход не равен тому, что должно. Вход отклоняется не более, чем на rho1, и не менее, чем на rho2
import subprocess
import time

import numpy as np
import tensorflow as tf
from larq.quantizers import SteSign
from pathlib import Path
import matplotlib.pyplot as plt

from src.encode_model.encode_model import Encode, Constraint
from src.encode_verifying.strange_input_by_output_for_mnistl import choose_reference_from_dataset


def save_img(image_, name):
    image_ = image_[:28 * 28 * 8]
    image_ = np.reshape(image_, (28, 28, 8))
    image = np.empty((28, 28))
    for i in range(len(image_)):
        for j in range(len(image_[0])):
            num = 0
            for n in range(len(image_[0][0])):
                if int(image_[i][j][n]) > 0:
                    num += 2 ** (8 - n - 1)
            image[i][j] = num

    # print(image)

    plt.imshow(image)
    plt.savefig(name + '.png')


def save_res_wrong(i):
    file_res = "../../solvers/res/res_input_by_output"

    file = open(file_res + str(i) + ".txt", 'r')
    if not file.readline():
        return
    res = file.readline()
    if not res:
        return
    vars = res.split()
    ans = ""

    for j in range(10):
        ans += "_" + str(int(int(vars[6400 + j]) > 0))

    save_img(vars, "../../solvers/img_res/res" + str(i) + ans)


# save_res_wrong(0)


def get_res_wrong(path, reference_input, output, rho1, rho2=0., is_strongly=False):
    print("path:", Path(path).stem, "!")
    if is_strongly:
        res_vars = input_by_output_strongly_wrong(path, reference_input, output, rho1, rho2)
    else:
        res_vars = input_by_output_wrong(path, reference_input, output, rho1, rho2)
    print("End coding", Path(path).stem)

    for i in range(15000):
        start = time.time()
        file_model = "../../data/CNFs/mnistl/input_by_output_" + Path(path).stem + ".cnfcc"
        file_res = "../../solvers/res/res_input_by_output" + str(i) + ".txt"
        subprocess.run(["./../../solvers/minisatcs/minisat", file_model, file_res])
        with open(file_model, "a") as myfile:
            file = open(file_res, 'r')
            if not file.readline():
                return
            res = file.readline()
            if not res:
                return

            vars = res.split()
            vars = list(map(int, vars))
            s = ""
            for j in range(28 * 28 * 8):
                s += str(-vars[j]) + " "
            s += "0\n"
            myfile.write("c appended cc " + str(i) + "\n" + s)
        end = time.time()
        print("The time of finding an adversarial example " + str(i) + ":", (end - start), "s")
        save_res_wrong(i)


def input_by_output_strongly_wrong(path, reference_input, output, rho1, rho2=0.):
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})
    encode = Encode(model, input_shape=tuple([28, 28, 8]))

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

    res_path = "../../data/CNFs/mnistl/input_by_output_" + Path(path).stem + ".cnfcc"
    encode.save(res_path)
    return encode.output_vars_layers[-1]


def input_by_output_wrong(path, reference_input, output, rho1, rho2=0.):
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})
    encode = Encode(model, input_shape=tuple([28, 28, 8]))

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

    res_path = "../../data/CNFs/mnistl/input_by_output_" + Path(path).stem + ".cnfcc"
    encode.save(res_path)
    return encode.output_vars_layers[-1]


ref, l = choose_reference_from_dataset()
# get_res_wrong("../../data/models/mnistl/model_from_pytorch.h5", ref, l, 0.3, is_strongly=False)
get_res_wrong("../../data/models/mnistl/model_from_pytorch2.h5", ref, l, 0.01, is_strongly=False)
