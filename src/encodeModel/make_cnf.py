import tensorflow as tf
from larq.quantizers import SteSign

from src.encodeModel.encode import Encode
from src.encodeModel.encode_constraint import encode_geq_and_leq, num2bits


def make_cnf_model(id):
    model = tf.keras.models.load_model("../../data/models/model" + str(id) + ".h5", custom_objects={
         "custom_activation": SteSign})  # "custom_activation" for layer Init
    encode = Encode(model.layers)
    encode.encode()
    encode.save_cnf("../../data/CNFs/model" + str(id) + ".cnf")


def make_cnf_with_input_constraint(id, a, b):
    model = tf.keras.models.load_model("../../data/models/model" + str(id) + ".h5",
                                       custom_objects={"custom_activation": SteSign})
    encode = Encode(model.layers)
    encode.encode()
    x = []
    for i in range(encode.output_vars_layers[0][1]):
        x.append(encode.output_vars_layers[0][0] + i)
    n = len(x)
    clauses = encode_geq_and_leq(x, num2bits(a, n), num2bits(b, n))
    encode.clauses.extend(clauses)
    encode.save_cnf("../../data/CNFs/model" + str(id) + "_input_constraint_" + str(a) + "_and_" + str(b) + ".cnf")


make_cnf_model(0)
make_cnf_with_input_constraint(0, 0, 100)
make_cnf_with_input_constraint(0, 100, 200)
make_cnf_with_input_constraint(0, 200, 300)
make_cnf_with_input_constraint(0, 300, 400)
make_cnf_with_input_constraint(0, 400, 500)
make_cnf_with_input_constraint(0, 500, 600)
make_cnf_with_input_constraint(0, 600, 700)
make_cnf_with_input_constraint(0, 700, 800)

make_cnf_model(4)
make_cnf_with_input_constraint(4, 0, 100)
make_cnf_with_input_constraint(4, 100, 200)
make_cnf_with_input_constraint(4, 200, 300)
make_cnf_with_input_constraint(4, 300, 400)
make_cnf_with_input_constraint(4, 400, 500)
make_cnf_with_input_constraint(4, 500, 600)
make_cnf_with_input_constraint(4, 600, 700)
make_cnf_with_input_constraint(4, 700, 800)
