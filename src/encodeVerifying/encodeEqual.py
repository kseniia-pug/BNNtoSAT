import tensorflow as tf
from larq.quantizers import SteSign

from src.encodeModel.encode import Encode, print_clauses


def make_cnf_equal_models(id1, id2):
    model1 = tf.keras.models.load_model("../../data/models/model" + str(id1) + ".h5",
                                        custom_objects={"custom_activation": SteSign})
    model2 = tf.keras.models.load_model("../../data/models/model" + str(id2) + ".h5",
                                        custom_objects={"custom_activation": SteSign})
    encode1 = Encode(model1.layers)
    encode1.encode()
    encode2 = Encode(model2.layers, id_start=encode1.all_vars[-1].id-encode1.output_vars_layers[0][1]+1)
    encode2.encode()
    encode2.change_input_vars(1)

    assert encode1.output_layers_shape[-1] == encode2.output_layers_shape[-1]
    assert encode1.output_vars_layers[-1][1] == encode2.output_vars_layers[-1][1]

    clauses = []
    res_clause = []
    for i in range(encode1.output_vars_layers[-1][1]):
        var = encode2.create_var('equal').id
        res_clause.append(var)
        clauses.append([encode1.output_vars_layers[-1][0] + i, encode2.output_vars_layers[-1][0] + i, -var])
        clauses.append([encode1.output_vars_layers[-1][0] + i, -(encode2.output_vars_layers[-1][0] + i), var])
        clauses.append([-(encode1.output_vars_layers[-1][0] + i), encode2.output_vars_layers[-1][0] + i, var])
        clauses.append([-(encode1.output_vars_layers[-1][0] + i), -(encode2.output_vars_layers[-1][0] + i), -var])
    clauses.append(res_clause)

    path = "../../data/CNFs/equal_models_" + str(id1) + "_and_" + str(id2) + ".cnf"
    file = open(path, 'w')
    file.write('p cnf ' + str(len(encode1.all_vars) + len(encode2.all_vars) - encode1.output_vars_layers[0][1]) + ' ' + str(len(encode1.clauses) + len(encode1.constraints) + len(encode2.clauses) + len(encode2.constraints) + len(clauses)) + '\n')
    print_clauses(clauses, file)
    file.close()

    encode1.save_cnf(path, mode='a')
    encode2.save_cnf(path, mode='a')


# make_cnf_equal_models(0, 0)
make_cnf_equal_models(0, 4)
make_cnf_equal_models(4, 0)
make_cnf_equal_models(4, 4)

make_cnf_equal_models(0, 5)
make_cnf_equal_models(5, 0)
make_cnf_equal_models(5, 5)

make_cnf_equal_models(5, 4)
make_cnf_equal_models(4, 5)
