from src.create_model.model import Model


def create_model0(dataset='mnist'):
    model = Model("model0")
    model.add_init()
    model.add_flatten()
    model.add_output(units=10)
    model.save(dataset)


def create_model100(dataset='mnist'):
    model = Model("model100")
    model.add_init()
    model.add_quant_conv2D((5, 5), strides=(1, 1), filters=6)
    model.add_flatten()
    model.add_output(units=10)
    model.save(dataset)


def create_model101(dataset='mnist'):
    model = Model("model101")
    model.add_init()
    model.add_quant_conv2D((5, 5), strides=(1, 1), filters=6)
    model.add_quant_conv2D((5, 5), strides=(1, 1), filters=12)
    model.add_quant_conv2D((3, 3), strides=(1, 1), filters=4)
    model.add_quant_conv2D((5, 5), strides=(1, 1), filters=1)
    model.add_flatten()
    model.add_output(units=10)
    model.save(dataset)


def create_model102(dataset='mnist'):
    model = Model("model102")
    model.add_init()
    model.add_pool2D_avg((2, 2), strides=(1, 1))
    model.add_flatten()
    model.add_output(units=10)
    model.save(dataset)


def create_model103(dataset='mnist'):
    model = Model("model103")
    model.add_init()
    model.add_quant_conv2D(kernel_size=(5, 5), filters=6)
    model.add_pool2D_avg((2, 2), strides=(2, 2))
    model.add_quant_conv2D(kernel_size=(5, 5), filters=16)
    model.add_pool2D_avg((2, 2), strides=(2, 2))
    model.add_flatten()
    model.add_output(units=120)
    model.add_output(units=84)
    model.add_output(units=10)
    model.save(dataset)


# def create_model1(dataset='mnist'):
#     model = Model("model1")
#     model.add_init()
#     model.add_quant_conv2D_maxpooling2D(kernel_size=32, strides=(3, 3), pool_size=(2, 2), input_shape=(28, 28, 1))
#     model.add_flatten()
#     model.add_output(units=10)
#     model.save(dataset)


# def create_model2(dataset='mnist'):
#     model = Model("model2")
#     model.add_init()
#     model.add_quant_conv2D_maxpooling2D(kernel_size=32, strides=(3, 3), pool_size=(2, 2), input_shape=(28, 28, 1))
#     model.add_flatten()
#     model.add_quant_dense(units=64)
#     model.add_output(units=10)
#     model.save(dataset)


# def create_model3(dataset='mnist'):
#     model = Model("model3")
#     model.add_init()
#     model.add_quant_conv2D_maxpooling2D(kernel_size=32, strides=(3, 3), pool_size=(2, 2), input_shape=(28, 28, 1))
#     model.add_quant_conv2D_maxpooling2D(kernel_size=64, strides=(3, 3), pool_size=(2, 2))
#     model.add_quant_conv2D(kernel_size=64, strides=(3, 3))
#     model.add_flatten()
#     model.add_quant_dense(units=64)
#     model.add_output(units=10)
#     model.save(dataset)


def create_model4(dataset='mnist'):
    model = Model("model4")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_output(units=10)
    model.save(dataset)


def create_model5(dataset='mnist'):
    model = Model("model5")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_output(units=10)
    model.save(dataset)


def create_model6(dataset='mnist'):  # 19
    model = Model("model6")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_output(units=10)
    model.save(dataset)


def create_model7(dataset='mnist'):  # 99
    model = Model("model7")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=2048)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=2048)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=2048)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=2048)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_output(units=10)
    model.save(dataset)


def create_model8(dataset='mnist'):  # 75, что-то не то с коэффициентами
    model = Model("model8")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=2048)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=2048)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=2048)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_output(units=10)
    model.save(dataset)


def create_model9(dataset='mnist'):  # 35
    model = Model("model9")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=2048)
    model.add_quant_dense(units=1854)
    model.add_quant_dense(units=1523)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=812)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_output(units=10)
    model.save(dataset)


def create_model10(dataset='mnist'):  # 51
    model = Model("model10")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=256)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=1024)
    model.add_quant_dense(units=512)
    model.add_quant_dense(units=128)
    model.add_quant_dense(units=64)
    model.add_quant_dense(units=32)
    model.add_quant_dense(units=21)
    model.add_output(units=10)
    model.save(dataset)


def create_model11(dataset='my'):
    model = Model("model11")
    model.add_init()
    model.add_flatten()
    model.add_output(units=2)
    model.save(dataset)


def create_model12(dataset='my'):
    model = Model("model12")
    model.add_init()
    model.add_flatten()
    model.add_quant_dense(units=32)
    model.add_output(units=2)
    model.save(dataset)


# create_model0()
# create_model1()
# create_model2()
# create_model3()
# create_model4()
# create_model5()
# create_model6()
# create_model7()
# create_model8()
# create_model9()
# create_model10()

# create_model0('cifar10')
# create_model1('cifar10')
# create_model2('cifar10')
# create_model3('cifar10')
# create_model4('cifar10')
# create_model5('cifar10')
# create_model6('cifar10')
# create_model7('cifar10')
# create_model8('cifar10')
# create_model9('cifar10')
# create_model10('cifar10')

# create_model0('cifar10l')
# create_model1('cifar10l')
# create_model2('cifar10l')
# create_model3('cifar10l')
# create_model4('cifar10l')
# create_model5('cifar10l')
# create_model6('cifar10l')
# create_model7('cifar10l')
# create_model8('cifar10l')
# create_model9('cifar10l')
# create_model10('cifar10l')

# create_model11()
# create_model12()

# create_model0('mnistl')
# create_model4('mnistl')
# create_model100()
# create_model101()
# create_model102()
# create_model103()

def create_model_from_pytorch(dataset='mnistl'):
    model = Model("model_from_pytorch")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_from_pytorch2(dataset='mnistl'):
    model = Model("model_from_pytorch2")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_for_struct2(dataset='mnistl'):
    model = Model("model_for_struct2")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)

    model.add_output(units=10)

    model.save(dataset)


# create_model_from_pytorch()
# create_model_from_pytorch2()

# create_model_for_struct2()


# ------------------------------------- test --------------------------------------------

def create_model_test1(dataset='mnistl'):
    model = Model("model_test1")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test2(dataset='mnistl'):
    model = Model("model_test2")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test3(dataset='mnistl'):
    model = Model("model_test3")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test4(dataset='mnistl'):
    model = Model("model_test4")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test5(dataset='mnistl'):
    model = Model("model_test5")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test6(dataset='mnistl'):
    model = Model("model_test6")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test7(dataset='mnistl'):
    model = Model("model_test7")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test8(dataset='mnistl'):
    model = Model("model_test8")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test9(dataset='mnistl'):
    model = Model("model_test9")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test10(dataset='mnistl'):
    model = Model("model_test10")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test11(dataset='mnistl'):
    model = Model("model_test11")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test12(dataset='mnistl'):
    model = Model("model_test12")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test13(dataset='mnistl'):
    model = Model("model_test13")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test14(dataset='mnistl'):
    model = Model("model_test14")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test15(dataset='mnistl'):
    model = Model("model_test15")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test16(dataset='mnistl'):
    model = Model("model_test16")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test17(dataset='mnistl'):
    model = Model("model_test17")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_quant_dense(units=64, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test18(dataset='mnistl'):
    model = Model("model_test18")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=2048, use_bias=True)
    model.add_quant_dense(units=1024, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=128, use_bias=True)
    model.add_quant_dense(units=64, use_bias=True)
    model.add_quant_dense(units=32, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test19(dataset='mnistl'):
    model = Model("model_test19")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=64, use_bias=True)
    model.add_quant_dense(units=64, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


def create_model_test20(dataset='mnistl'):
    model = Model("model_test20")

    model.add_init()
    model.add_flatten()

    model.add_quant_dense(units=4096, use_bias=True)
    model.add_quant_dense(units=512, use_bias=True)
    model.add_quant_dense(units=64, use_bias=True)
    model.add_quant_dense(units=16, use_bias=True)
    model.add_output(units=10)

    model.save(dataset)


create_model_test1()
create_model_test2()
create_model_test3()
create_model_test4()
create_model_test5()
create_model_test6()
create_model_test7()
create_model_test8()
create_model_test9()
create_model_test10()
create_model_test11()
create_model_test12()
create_model_test13()
create_model_test14()
create_model_test15()
create_model_test16()
create_model_test17()
create_model_test18()
create_model_test19()
create_model_test20()
