import tensorflow as tf
import larq as lq

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

    def add_quant_conv2D_maxpooling2D(self, kernel_size, strides, pool_size, input_shape=None, use_bias=False, scale=False):
        scale = True  # TODO
        if input_shape != None:
            self.model.add(lq.layers.QuantConv2D(kernel_size, strides, use_bias=use_bias, input_shape=input_shape,
                                            name=str(self.ind) + "/QuantConv2DMaxPooling2D/QuantConv2D", **self.kwargs))
        else:
            self.model.add(lq.layers.QuantConv2D(kernel_size, strides, use_bias=use_bias,
                                            name=str(self.ind) + "/QuantConv2DMaxPooling2D/QuantConv2D", **self.kwargs))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size, name=str(self.ind) + "/QuantConv2DMaxPooling2D/MaxPooling2D"))
        self.model.add(tf.keras.layers.BatchNormalization(scale=scale,
                                                     name=str(self.ind) + "/QuantConv2DMaxPooling2D/BatchNormalization"))
        self.model.add(tf.keras.layers.Activation("ste_sign", name=str(self.ind) + "/QuantConv2DMaxPooling2D/Activation"))
        self.ind += 1

    def add_quant_conv2D(self, kernel_size, strides, input_shape=None, use_bias=False, scale=False):
        scale = True  # TODO
        if input_shape != None:
            self.model.add(lq.layers.QuantConv2D(kernel_size, strides, use_bias=use_bias, input_shape=input_shape,
                                            name=str(self.ind) + "/QuantConv2D/QuantConv2D", **self.kwargs))
        else:
            self.model.add(lq.layers.QuantConv2D(kernel_size, strides, use_bias=use_bias,
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

    def add_output(self, units, use_bias=False):
        self.model.add(lq.layers.QuantDense(units, use_bias=use_bias, name=str(self.ind) + "/Output/QuantDense", **self.kwargs))
        self.model.add(tf.keras.layers.Activation("softmax", name=str(self.ind) + "/Output/Activation"))
        self.ind += 1

    def save(self):
        # Load MNIST
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1  # Normalize pixel values to be between -1 and 1

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(train_images, train_labels, batch_size=64, epochs=5)
        _, test_acc = self.model.evaluate(test_images, test_labels)
        print(f"Test accuracy {test_acc * 100:.2f} %")
        with lq.context.quantized_scope(True):
            self.model.save("../../data/models/" + self.name + ".h5")
