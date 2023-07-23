# Это просто пример кодировки сети, решение этих формул ничего не говорит об их устройстве

import tensorflow as tf
from larq.quantizers import SteSign
from pathlib import Path

from src.encode_model.encode_model import Encode


def encode_model(path):
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})  # "custom_activation" for layer Init
    encode = Encode(model)
    encode.save("../../data/CNFs/" + Path(path).stem + ".cnfcc")


encode_model("../../data/models/model0.h5")
encode_model("../../data/models/model4.h5")
encode_model("../../data/models/model5.h5")