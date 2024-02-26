# Это просто пример кодировки сети, решение этих формул ничего не говорит об их устройстве

import tensorflow as tf
from larq.quantizers import SteSign
from pathlib import Path

from src.encode_model.encode_model import Encode


def encode_model(path, input_shape=tuple([28, 28, 8])):
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})  # "custom_activation" for layer Init
    encode = Encode(model, input_shape=input_shape)
    encode.save("../../data/CNFs/" + Path(path).parents[0].stem + "/" + Path(path).stem + ".cnfcc")


# encode_model("../../data/models/mnist/model0.h5")
# encode_model("../../data/models/mnist/model4.h5")
# encode_model("../../data/models/mnist/model5.h5")
# encode_model("../../data/models/mnist/model6.h5")
# encode_model("../../data/models/mnist/model7.h5")

# encode_model("../../data/models/my/model12.h5", input_shape=tuple([4]))
# encode_model("../../data/models/mnist/model100.h5")
encode_model("../../data/models/mnistl/model_from_pytorch.h5")
