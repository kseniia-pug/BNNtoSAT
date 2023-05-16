import os

import tensorflow as tf
import larq as lq
from src.createModel.layers import addInit, addFlatten, addOutput, addQuantConv2DMaxPooling2D, addQuantDense, addQuantConv2D


def CreateModel0(train_images, train_labels, test_images, test_labels):
    model = tf.keras.models.Sequential()
    ind = [0]

    addInit(model, ind)
    addFlatten(model, ind)
    addOutput(model, 10, ind)

    SaveModel(model, train_images, train_labels, test_images, test_labels, "model0.h5")

def CreateModel1(train_images, train_labels, test_images, test_labels):
    model = tf.keras.models.Sequential()
    ind = [0]

    addInit(model, ind)
    addQuantConv2DMaxPooling2D(model, 32, (3, 3), (2, 2), ind, input_shape=(28, 28, 1))
    addFlatten(model, ind)
    addOutput(model, 10, ind)

    SaveModel(model, train_images, train_labels, test_images, test_labels, "model1.h5")

def CreateModel2(train_images, train_labels, test_images, test_labels):
    model = tf.keras.models.Sequential()
    ind = [0]

    addInit(model, ind)
    addQuantConv2DMaxPooling2D(model, 32, (3, 3), (2, 2), ind, input_shape=(28, 28, 1))
    addFlatten(model, ind)
    addQuantDense(model, 64, ind)
    addOutput(model, 10, ind)

    SaveModel(model, train_images, train_labels, test_images, test_labels, "model2.h5")

def CreateModel3(train_images, train_labels, test_images, test_labels):
    model = tf.keras.models.Sequential()
    ind = [0]

    addInit(model, ind)
    addQuantConv2DMaxPooling2D(model, 32, (3, 3), (2, 2), ind, input_shape=(28, 28, 1))
    addQuantConv2DMaxPooling2D(model, 64, (3, 3), (2, 2), ind)
    addQuantConv2D(model, 64, (3, 3), ind)
    addFlatten(model, ind)
    addQuantDense(model, 64, ind)
    addOutput(model, 10, ind)

    SaveModel(model, train_images, train_labels, test_images, test_labels, "model3.h5")

def CreateModel4(train_images, train_labels, test_images, test_labels):
    model = tf.keras.models.Sequential()
    ind = [0]

    addInit(model, ind)
    addFlatten(model, ind)
    addQuantDense(model, 32, ind)
    addOutput(model, 10, ind)

    SaveModel(model, train_images, train_labels, test_images, test_labels, "model4.h5")

def CreateModel5(train_images, train_labels, test_images, test_labels):
    model = tf.keras.models.Sequential()
    ind = [0]

    addInit(model, ind)
    addFlatten(model, ind)
    addQuantDense(model, 32, ind)
    addQuantDense(model, 64, ind)
    addQuantDense(model, 128, ind)
    addOutput(model, 10, ind)

    SaveModel(model, train_images, train_labels, test_images, test_labels, "model5.h5")

def SaveModel(model, train_images, train_labels, test_images, test_labels, name):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, batch_size=64, epochs=5)
    _, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy {test_acc * 100:.2f} %")
    with lq.context.quantized_scope(True):
        model.save("../../data/models/" + name)
