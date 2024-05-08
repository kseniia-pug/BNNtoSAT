import larq.layers
import tensorflow as tf
import larq as lq
import numpy as np

# larq.layers.QuantDense(k_otput, use_bias=True/False) -- полносвязный слой
# larq.layers.QuantConv2D(k_otput, (kernel_h, kernel_w), use_bias=True/False) -- 2D свёртка
# tf.keras.layers.MaxPooling2D((pool_h, pool_w)) -- 2D свёртка
# tf.keras.layers.BatchNormalization() -- gamma * (batch - moving_mean) / sqrt(moving_variance + epsilon) + beta, batch -- текущее значение
# tf.keras.layers.Flatten() -- в одну плоскость все размазывает
# model.add(tf.keras.layers.Activation("softmax")) -- exp(x) / sum(exp(x))

class Model:
    def __init__(self, name):
        self.name = name
        self.model = tf.keras.models.Sequential()
        self.ind = 0
        self.kwargs = dict(input_quantizer="ste_sign",
                  kernel_quantizer="ste_sign",
                  kernel_constraint=lq.constraints.weight_clip(clip_value=1))

    def add_init(self):
        self.model.add(tf.keras.layers.Activation("ste_sign", name=str(self.ind) + "/Init/Activation"))
        self.ind += 1

    def add_quant_conv2D(self, kernel_size, strides=(1, 1), filters=(1, 1), input_shape=None, use_bias=False, scale=False):
        scale = True  # TODO
        if input_shape != None:
            self.model.add(lq.layers.QuantConv2D(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, input_shape=input_shape,
                                            name=str(self.ind) + "/QuantConv2D/QuantConv2D", **self.kwargs))
        else:
            self.model.add(lq.layers.QuantConv2D(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias,
                                            name=str(self.ind) + "/QuantConv2D/QuantConv2D", **self.kwargs))
        self.model.add(tf.keras.layers.BatchNormalization(scale=scale, name=str(self.ind) + "/QuantConv2D/BatchNormalization"))
        self.model.add(tf.keras.layers.Activation("ste_sign", name=str(self.ind) + "/QuantConv2D/Activation"))
        self.ind += 1

    def add_flatten(self):
        self.model.add(tf.keras.layers.Flatten(name=str(self.ind) + "/Flatten/Flatten"))
        self.ind += 1


    def add_quant_dense(self, units, use_bias=False, scale=False):
        scale = True  # TODO
        self.model.add(lq.layers.QuantDense(units, use_bias=use_bias, name=str(self.ind) + "/QuantDense/QuantDense", **self.kwargs))
        self.model.add(tf.keras.layers.BatchNormalization(scale=scale, name=str(self.ind) + "/QuantDense/BatchNormalization"))
        self.model.add(tf.keras.layers.Activation("ste_sign", name=str(self.ind) + "/QuantDense/Activation"))
        self.ind += 1

    def add_pool2D_avg(self, pool_size, strides=(1, 1), scale=False):
        scale = True  # TODO
        self.model.add(tf.keras.layers.AveragePooling2D(pool_size, strides, name=str(self.ind) + "/Pool2DAvg/Pool2DAvg"))
        self.model.add(
            tf.keras.layers.BatchNormalization(scale=scale, name=str(self.ind) + "/Pool2DAvg/BatchNormalization"))
        self.model.add(tf.keras.layers.Activation("ste_sign", name=str(self.ind) + "/Pool2DAvg/Activation"))
        self.ind += 1

    def add_output(self, units, use_bias=False):
        self.model.add(lq.layers.QuantDense(units, use_bias=use_bias, name=str(self.ind) + "/Output/QuantDense", **self.kwargs))
        self.model.add(tf.keras.layers.Activation("softmax", name=str(self.ind) + "/Output/Activation"))
        self.ind += 1

    def save(self, dataset='mnist'):
        if dataset == 'mnist':
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
            train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1  # Normalize pixel values to be between -1 and 1
            train_images = train_images.reshape(-1, 28, 28, 1)
            test_images = test_images.reshape(-1, 28, 28, 1)
        elif dataset == 'mnistl':
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
                    print(str(k) + "/" + str(len(train_images_)))

            train_images = np.empty((len(train_images_), 28, 28, 8))
            for k in range(len(train_images_)):
                for i in range(len(train_images_[0])):
                    for j in range(len(train_images_[0][0])):
                        num = '{0:08b}'.format(train_images_[k][i][j])
                        for n in range(len(num)):
                            if num[n] == '1':
                                train_images[k][i][j][n] = 1
                            else:
                                train_images[k][i][j][n] = -1
                if k % 100 == 0:
                    print(str(k) + "/" + str(len(train_images_)))
            # print(test_images_[:10])
            # print(test_images[:10])
        elif dataset == 'cifar10':
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
            train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1  # Normalize pixel values to be between -1 and 1
        elif dataset == 'cifar10l':
            (train_images_, train_labels), (test_images_, test_labels) = tf.keras.datasets.cifar10.load_data()
            train_images_ = train_images_[:100] # TODO
            test_images_ = test_images_[:100] # TODO
            train_labels = train_labels[:100]  # TODO
            test_labels = test_labels[:100]  # TODO

            test_images = np.empty(test_images_.shape + (8,))
            for i in range(len(test_images_)):
                for j in range(len(test_images_[0])):
                    for k in range(len(test_images_[0][0])):
                        for l in range(len(test_images_[0][0][0])):
                            num = '{0:08b}'.format(test_images_[i][j][k][l])
                            for n in range(len(num)):
                                if num[n] == '1':
                                    test_images[i][j][k][l][n] = 1
                                else:
                                    test_images[i][j][k][l][n] = -1

            train_images = np.empty(train_images_.shape + (8,))
            for i in range(len(train_images_)):
                for j in range(len(train_images_[0])):
                    for k in range(len(train_images_[0][0])):
                        for l in range(len(train_images_[0][0][0])):
                            num = '{0:08b}'.format(train_images_[i][j][k][l])
                            for n in range(len(num)):
                                if num[n] == '1':
                                    train_images[i][j][k][l][n] = 1
                                else:
                                    train_images[i][j][k][l][n] = -1
        else:
            train_images = [[-1., -1., -1., -1.], [-1., 1., -1., -1.], [-1., -1., -1., 1.], [1., -1., 1., -1.], [-1., 1., 1., -1.], [-1., -1., 1., 1.], [1., 1., -1., 1.], [1., -1., 1., 1.]]
            train_labels = [0, 0, 0, 1, 0, 1, 1, 1]
            test_images = [[1., -1., -1., -1.], [-1., -1., 1., -1.], [1., 1., -1., -1.], [1., -1., -1., 1.], [-1., 1., -1., 1.], [1., 1., 1., -1.], [-1., 1., 1., 1.], [1., 1., 1., 1.]]
            test_labels = [0, 0, 1, 0, 1, 1, 1, 1]

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(train_images, train_labels, batch_size=64, epochs=20)
        _, test_acc = self.model.evaluate(test_images, test_labels)
        print(f"Test accuracy {test_acc * 100:.2f} %")
        with lq.context.quantized_scope(True):
            self.model.save("../../data/models/" + dataset + "/" + self.name + ".h5")
