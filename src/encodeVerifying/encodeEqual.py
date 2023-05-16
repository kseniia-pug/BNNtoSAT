import tensorflow as tf
from larq.quantizers import SteSign

from src.encodeModel.encode import Encode


def make_cnf_equal_models(id1, id2):
    model1 = tf.keras.models.load_model("../../data/models/model" + str(id1) + ".h5",
                                        custom_objects={"custom_activation": SteSign})
    model2 = tf.keras.models.load_model("../../data/models/model" + str(id2) + ".h5",
                                        custom_objects={"custom_activation": SteSign})
    encode1 = Encode(model1.layers)
    encode1.encode()
    encode2 = Encode(model2.layers, id_start=encode1.all_vars[-1].id + 1)
    encode2.encode()
    encode2.change_input_vars(1)

    assert encode1.output_layers_shape[-1] == encode2.output_layers_shape[-1]
    assert encode1.output_vars_layers[-1][1] == encode2.output_vars_layers[-1][1]

    clauses = []
    for i in range(encode1.output_vars_layers[-1][1]):
        clauses.append([encode1.output_vars_layers[-1][0] + i, encode2.output_vars_layers[-1][0] + i])
        clauses.append([-(encode1.output_vars_layers[-1][0] + i), -(encode2.output_vars_layers[-1][0] + i)])
    encode1.clauses.extend(clauses)

    path = "../../data/CNFs/equal_models_" + str(id1) + "_and_" + str(id2) + ".cnf"
    encode1.save_cnf(path)
    encode2.save_cnf(path, mode='a')


make_cnf_equal_models(0, 0)
make_cnf_equal_models(0, 4)
make_cnf_equal_models(4, 0)
make_cnf_equal_models(4, 4)

make_cnf_equal_models(0, 5)
make_cnf_equal_models(5, 0)
make_cnf_equal_models(5, 5)

make_cnf_equal_models(5, 4)
make_cnf_equal_models(4, 5)
