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


make_cnf_model(0)
make_cnf_model(4)
make_cnf_model(5)
