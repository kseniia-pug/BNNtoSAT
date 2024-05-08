# странный вход для данного выхода
# выход равен тому, что должно. Вход отклоняется более, чем на rho1, и не более, чем на rho2
import subprocess
import random

import numpy as np
import tensorflow as tf
from larq.quantizers import SteSign
from pathlib import Path
import matplotlib.pyplot as plt

from src.encode_model.encode_model import Encode, Constraint

np.set_printoptions(threshold=20000)
# (train_images_, train_labels), _ = tf.keras.datasets.mnist.load_data()
# print(train_images_[0])
# print(train_labels[0])


def choose_reference_from_dataset(name=""):
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
    id = random.randint(0, len(train_images) - 1)

    # print(train_images[id])
    # print(train_labels[id])

    train_image_ = train_images[id]
    train_image = np.empty((28, 28, 8))
    for i in range(len(train_image_)):
        for j in range(len(train_image_[0])):
            num = '{0:08b}'.format(train_image_[i][j])
            for n in range(len(num)):
                if num[n] == '1':
                    train_image[i][j][n] = 1
                else:
                    train_image[i][j][n] = -1

    train_label = np.zeros(10)
    train_label[train_labels[id]] = 1

    ans = ""
    for j in range(10):
        ans += "_" + str(int(train_label[j]))

    plt.imshow(train_image_)
    plt.savefig('../../solvers/img_res/reference_' + name + '_' + ans + '.png')

    # print(train_image)
    # print(train_label)

    return train_image, train_label


def print_img(image_):
    image = np.empty((28, 28))
    for i in range(len(image_)):
        for j in range(len(image_[0])):
            num = 0
            for n in range(len(image_[0][0])):
                if image_[i][j][n] == 1:
                    num += 2 ** (8-n-1)
            image[i][j] = num

    # print(image)

    plt.imshow(image)
    plt.show()


# im, l = choose_reference_from_dataset()
# print_img(im)


def print_res_strange(path, reference_input, output, rho1, rho2=1., is_strongly=False):
    if is_strongly:
        res_vars = input_by_output_strongly_strange(path, reference_input, output, rho1, rho2)
    else:
        res_vars = input_by_output_strange(path, reference_input, output, rho1, rho2)
    print("End coding")

    subprocess.run(
        ["./../../solvers/minisatcs/minisat", "../../data/CNFs/mnistl/input_by_output_" + Path(path).stem + ".cnfcc",
         "../../solvers/res/res_input_by_output.txt"])
    file = open("../../solvers/res/res_input_by_output.txt", 'r')
    if not file.readline():
        return
    res = file.readline()
    if not res:
        return

    # print_img(res)
    # pixels = res.split()
    # pixels = list(map(int, pixels))
    #
    # print([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(list(map(lambda num: int(num > 0), pixels[res_vars[0] - 1:res_vars[0] - 1 + res_vars[1]])))
    # print(list(map(lambda num: abs(int(num)), pixels[res_vars[0] - 1:res_vars[0] - 1 + res_vars[1]])))

    # pixels = pixels[:28 * 28]
    #
    # new_pixels = []
    # for pixel in pixels:
    #     if pixel < 0:
    #         new_pixels.append('█')
    #     else:
    #         new_pixels.append(' ')
    #
    # for i in range(len(new_pixels)):
    #     if i % 28 == 0:
    #         print('\n', end='')
    #     print(new_pixels[i], end='')


def input_by_output_strongly_strange(path, reference_input, output, rho1, rho2=1.):
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})
    encode = Encode(model, input_shape=tuple([28, 28, 8]))

    assert encode.output_layers_shape[-1] == output.shape
    # print(encode.input_shape, reference_input.shape)
    assert encode.input_shape == reference_input.shape

    reference_input = reference_input.flatten()
    output = output.flatten()

    for i in range(len(output)):
        print(encode.output_vars_layers[-1][0] + i)
        if output[i] == 1:
            encode.clauses.append([encode.output_vars_layers[-1][0] + i])
        else:
            encode.clauses.append([-(encode.output_vars_layers[-1][0] + i)])

    constraint = Constraint()
    res = encode.create_var('variance').id
    encode.clauses.append([res])
    constraint.res = res
    constraint.c = int(len(reference_input) * rho1)
    for i in range(len(reference_input)):
        if reference_input[i] == 1:
            constraint.vars.append(-(encode.output_vars_layers[0][0] + i))
        else:
            constraint.vars.append(encode.output_vars_layers[0][0] + i)
    encode.constraints.append(constraint)

    constraint = Constraint()
    res = encode.create_var('variance').id
    encode.clauses.append([res])
    constraint.res = res
    constraint.c = int(len(reference_input) * (1 - rho2))
    for i in range(len(reference_input)):
        if reference_input[i] == 1:
            constraint.vars.append(encode.output_vars_layers[0][0] + i)
        else:
            constraint.vars.append(-(encode.output_vars_layers[0][0] + i))
    encode.constraints.append(constraint)

    res_path = "../../data/CNFs/mnistl/input_by_output_" + Path(path).stem + ".cnfcc"
    encode.save(res_path)
    return encode.output_vars_layers[-1]


def input_by_output_strange(path, reference_input, output, rho1, rho2=1.):
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})
    encode = Encode(model, input_shape=tuple([28, 28, 8]))

    assert encode.output_layers_shape[-1] == output.shape
    # print(encode.input_shape, reference_input.shape)
    assert encode.input_shape == reference_input.shape

    reference_input = reference_input.flatten()
    output = output.flatten()

    for i in range(len(output)):
        if output[i] == 1:
            encode.clauses.append([encode.output_vars_layers[-1][0] + i])

    constraint = Constraint()
    res = encode.create_var('variance').id
    encode.clauses.append([res])
    constraint.res = res
    constraint.c = int(len(reference_input) * rho1)
    for i in range(len(reference_input)):
        if reference_input[i] == 1:
            constraint.vars.append(-(encode.output_vars_layers[0][0] + i))
        else:
            constraint.vars.append(encode.output_vars_layers[0][0] + i)
    encode.constraints.append(constraint)

    constraint = Constraint()
    res = encode.create_var('variance').id
    encode.clauses.append([res])
    constraint.res = res
    constraint.c = int(len(reference_input) * (1 - rho2))
    for i in range(len(reference_input)):
        if reference_input[i] == 1:
            constraint.vars.append(encode.output_vars_layers[0][0] + i)
        else:
            constraint.vars.append(-(encode.output_vars_layers[0][0] + i))
    encode.constraints.append(constraint)

    res_path = "../../data/CNFs/mnistl/input_by_output_" + Path(path).stem + ".cnfcc"
    encode.save(res_path)
    return encode.output_vars_layers[-1]


# ref, l = choose_reference_from_dataset()
# print_res_strange("../../data/models/mnistl/model_from_pytorch.h5", ref, l, 0., is_strongly=False)
