import tensorflow as tf
import larq as lq

# larq.layers.QuantDense(k_otput, use_bias=True/False) -- полносвязный слой
# larq.layers.QuantConv2D(k_otput, (kernel_h, kernel_w), use_bias=True/False) -- 2D свёртка
# tf.keras.layers.MaxPooling2D((pool_h, pool_w)) -- 2D свёртка
# tf.keras.layers.BatchNormalization() -- gamma * (batch - moving_mean) / sqrt(moving_variance + epsilon) + beta, batch -- текущее значение
# tf.keras.layers.Flatten() -- в одну плоскость все размазывает
# model.add(tf.keras.layers.Activation("softmax")) -- exp(x) / sum(exp(x))

kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint=lq.constraints.weight_clip(clip_value=1))


def addInit(model, ind):
    model.add(tf.keras.layers.Activation("ste_sign", name=str(ind[0]) + "/Init/Activation"))
    ind[0] += 1


def addQuantConv2DMaxPooling2D(model, kernel_size, strides, pool_size, ind, input_shape=None, use_bias=False,
                               scale=False):
    scale = True  # TODO
    if input_shape != None:
        model.add(lq.layers.QuantConv2D(kernel_size, strides, use_bias=use_bias, input_shape=input_shape,
                                        name=str(ind[0]) + "/QuantConv2DMaxPooling2D/QuantConv2D", **kwargs))
    else:
        model.add(lq.layers.QuantConv2D(kernel_size, strides, use_bias=use_bias,
                                        name=str(ind[0]) + "/QuantConv2DMaxPooling2D/QuantConv2D", **kwargs))
    model.add(tf.keras.layers.MaxPooling2D(pool_size, name=str(ind[0]) + "/QuantConv2DMaxPooling2D/MaxPooling2D"))
    model.add(tf.keras.layers.BatchNormalization(scale=scale,
                                                 name=str(ind[0]) + "/QuantConv2DMaxPooling2D/BatchNormalization"))
    model.add(tf.keras.layers.Activation("ste_sign", name=str(ind[0]) + "/QuantConv2DMaxPooling2D/Activation"))
    ind[0] += 1


def addQuantConv2D(model, kernel_size, strides, ind, input_shape=None, use_bias=False, scale=False):
    scale = True  # TODO
    if input_shape != None:
        model.add(lq.layers.QuantConv2D(kernel_size, strides, use_bias=use_bias, input_shape=input_shape,
                                        name=str(ind[0]) + "/QuantConv2D/QuantConv2D", **kwargs))
    else:
        model.add(lq.layers.QuantConv2D(kernel_size, strides, use_bias=use_bias,
                                        name=str(ind[0]) + "/QuantConv2D/QuantConv2D", **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=scale, name=str(ind[0]) + "/QuantConv2D/BatchNormalization"))
    model.add(tf.keras.layers.Activation("ste_sign", name=str(ind[0]) + "/QuantConv2D/Activation"))
    ind[0] += 1


def addFlatten(model, ind):
    model.add(tf.keras.layers.Flatten(name=str(ind[0]) + "/Flatten/Flatten"))
    ind[0] += 1


def addQuantDense(model, units, ind, use_bias=False, scale=False):
    scale = True  # TODO
    model.add(lq.layers.QuantDense(units, use_bias=use_bias, name=str(ind[0]) + "/QuantDense/QuantDense", **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=scale, name=str(ind[0]) + "/QuantDense/BatchNormalization"))
    model.add(tf.keras.layers.Activation("ste_sign", name=str(ind[0]) + "/QuantDense/Activation"))
    ind[0] += 1


def addOutput(model, units, ind, use_bias=False):
    model.add(lq.layers.QuantDense(units, use_bias=use_bias, name=str(ind[0]) + "/Output/QuantDense", **kwargs))
    model.add(tf.keras.layers.Activation("softmax", name=str(ind[0]) + "/Output/Activation"))
    ind[0] += 1
