from src.create_model.model import Model


def create_model0():
    model = Model("model0")
    model.add_init()
    model.add_flatten()
    model.add_output(units=10)
    model.save()


def create_model1():
    model = Model("model1")
    model.add_init()
    model.add_quant_conv2D_maxpooling2D(kernel_size=32, strides=(3, 3), pool_size=(2, 2), input_shape=(28, 28, 1))
    model.add_flatten()
    model.add_output(units=10)
    model.save()


def create_model2():
    model = Model("model2")
    model.add_init()
    model.add_quant_conv2D_maxpooling2D(kernel_size=32, strides=(3, 3), pool_size=(2, 2), input_shape=(28, 28, 1))
    model.add_flatten()
    model.add_quant_dense(units=64)
    model.add_output(units=10)
    model.save()


def create_model3():
    model = Model("model3")
    model.add_init()
    model.add_quant_conv2D_maxpooling2D(kernel_size=32, strides=(3, 3), pool_size=(2, 2), input_shape=(28, 28, 1))
    model.add_quant_conv2D_maxpooling2D(kernel_size=64, strides=(3, 3), pool_size=(2, 2))
    model.add_quant_conv2D(kernel_size=64, strides=(3, 3))
    model.add_flatten()
    model.add_quant_dense(units=64)
    model.add_output(units=10)
    model.save()


def create_model4():
    model = Model("model4")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_output(units=10)
    model.save()


def create_model5():
    model = Model("model5")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_output(units=10)
    model.save()


create_model0()
create_model1()
create_model2()
create_model3()
create_model4()
create_model5()
