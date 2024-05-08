# Рандомный вход, который отклоняется не более, чем на rho1, и не менее, чем на rho2
import subprocess
import time
import larq as lq
import numpy as np
import tensorflow as tf
from larq.quantizers import SteSign
from pathlib import Path
import matplotlib.pyplot as plt

from src.create_model.model import Model
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


def save_res_wrong(path, name):
    file = open(path)
    if not file.readline():
        return
    res = file.readline()
    if not res:
        return
    vars = res.split()
    ans = ""

    for j in range(10):
        ans += "_" + str(int(int(vars[6400 + j]) > 0))

    save_img(vars, "../../solvers/random/img_res/res" + str(i) + ans + "_" + name)


# save_res_wrong(0)


def get_res_wrong(path, reference_input, output, rho1, rho2=0., name=""):
    print("path:", Path(path).stem, "!")
    res_vars = input_by_output_wrong(path, reference_input, output, rho1, rho2)
    print("End coding", Path(path).stem, res_vars)

    for i in range(1005):
        start = time.time()
        file_model = "../../data/CNFs/mnistl/input_by_output_" + Path(path).stem + ".cnfcc"
        file_res = "../../solvers/random/res/res_input_by_output_" + name + "_" + str(i) + ".txt"
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
        with open("../../solvers/random/res.txt", "a") as myfile:
            myfile.write(str(end - start) + "\n")
        save_res_wrong(file_res, "res_input_by_output_" + name + "_" + str(i))


def input_by_output_wrong(path, reference_input, output, rho1, rho2=0.):
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})
    encode = Encode(model, input_shape=tuple([28, 28, 8]))

    assert encode.output_layers_shape[-1] == output.shape
    assert encode.input_shape == reference_input.shape

    reference_input = reference_input.flatten()
    output = output.flatten()

    # for i in range(len(output)):
    #     if output[i] == 1:
    #         encode.clauses.append([-(encode.output_vars_layers[-1][0] + i)])

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


(train_images_, train_labels), (test_images_, test_labels) = tf.keras.datasets.mnist.load_data()
train_images_ = train_images_[:10000]  # TODO
test_images_ = test_images_[:1000]  # TODO
train_labels = train_labels[:10000]  # TODO
test_labels = test_labels[:1000]  # TODO

test_images = np.empty((len(test_images_), 28, 28, 8))
for k in range(len(test_images_)):
    for i in range(len(test_images_[0])):
        for j in range(len(test_images_[0][0])):
            num = '{0:08b}'.format(test_images_[k][i][j])
            for n in range(len(num)):
                if num[n] == '1':
                    test_images[k][i][j][n] = 1
                else:
                    test_images[k][i][j][n] = -1
    if k % 100 == 0:
        # break
        print(str(k) + "/" + str(len(test_images_)))

print(train_images_.shape)
train_images = np.empty((len(train_images_) + 14*5, 28, 28, 8))
for i in range(len(train_images_)):
    for j in range(len(train_images_[0])):
        for k in range(len(train_images_[0][0])):
            num = '{0:08b}'.format(train_images_[i][j][k])
            for n in range(len(num)):
                if num[n] == '1':
                    train_images[i][j][k][n] = 1
                else:
                    train_images[i][j][k][n] = -1
    if i % 100 == 0:
        # break
        print(str(i) + "/" + str(len(train_images_)))

# print(train_images.shape)
# labels = [2, 2, 0, 0, 9, 5, 4, 4, 1, 0, 1, 5, 9, 9]
# for t in range(14):
#     for l in range(5):
#         if t == 13 and l > 81:
#             break
#         file = open("../../solvers/res/res_input_by_output_" + str(t) + "_" + str(l) + ".txt")
#         if not file.readline():
#             continue
#         res = file.readline()
#         if not res:
#             continue
#         vars = res.split()
#
#         vars = vars[:28 * 28 * 8]
#         vars = np.reshape(vars, (28, 28, 8))
#         image = np.empty((28, 28, 8))
#         for i in range(len(vars)):
#             for j in range(len(vars[0])):
#                 for n in range(len(vars[0][0])):
#                     if int(vars[i][j][n]) > 0:
#                         image[i][j][n] = 1.
#                     else:
#                         image[i][j][n] = -1.
#
#         # length = train_images.shape[0]
#         print(image.shape, train_images.shape)
#         # train_images = np.append(train_images, image)
#         train_images[len(train_images_)+t*l] = image
#         # train_images.reshape((length + 1, 28, 28, 8))
#         print(image.shape, train_images.shape)
#         train_labels = np.append(train_labels, labels[t])
#
#         if l % 10 == 0:
#             print(str(t) + ": " + str(l) + "/100")
#             print(len(train_images), len(train_labels))
#
#
# model = Model("model_from_pytorch2_after")
# model.add_init()
# model.add_flatten()
# model.add_quant_dense(units=128, use_bias=True)
# model.add_output(units=10)

# model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.model.fit(train_images, train_labels, batch_size=64, epochs=10)
# _, test_acc = model.model.evaluate(test_images, test_labels)
# print(f"Test accuracy {test_acc * 100:.2f} %")
# with lq.context.quantized_scope(True):
#     model.model.save("../../data/models/mnitl/input_by_output_model_from_pytorch2_after.h5")

for i in range(1):
    # if i < 3:
    #     continue
    ref, l = choose_reference_from_dataset(name=str(i))
    # get_res_wrong("../../data/models/mnistl/model_from_pytorch.h5", ref, l, 0.3, is_strongly=False)
    get_res_wrong("../../data/models/mnistl/model_from_pytorch2.h5", ref, l, 0.03, name=str(i))
