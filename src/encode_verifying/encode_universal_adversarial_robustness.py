import random
import subprocess
from pathlib import Path

import tensorflow as tf
from larq.quantizers import SteSign

from src.encode_model.encode_model import Encode
from src.encode_verifying.encode_universal_adversarial_robustness_with_tau import encode_universal_adversarial_robustness_with_tau


# SAT -- неправильно классифицирована хотя бы rho-ая доля при случайном tau
def encode_universal_adversarial_robustness(path, e, rho):  # tau: [|input|], e: 0..1 -- процент единиц в tau, rho: 0..1
    model = tf.keras.models.load_model(path, custom_objects={"custom_activation": SteSign})
    encode = Encode(model)
    encode_with_tau = Encode(model, id_start=encode.all_vars[-1].id + 1)

    tau = [0] * encode_with_tau.output_vars_layers[0][1]
    for i in range(len(tau)):
        tau[i] = random.choices([0, 1], weights=[1-e, e])[0]
    print("tau = ", tau)

    encode_universal_adversarial_robustness_with_tau(path, tau, rho)


def percent_of_robustness(path, e, rho, k=20):
    unsat = 0
    for _ in range(k):
        encode_universal_adversarial_robustness(path, e, rho)
        res = subprocess.run(["./../../solvers/minisatcs/minisat", "../../data/CNFs/universal_adversarial_robustness_" + Path(path).stem + ".cnfcc"], capture_output=True)
        for s in str(res.stdout).split('\\n'):
            if s[:2] == "r=":
                if s[2:] == "UNSAT":
                    unsat += 1
    print(unsat / k)


percent_of_robustness("../../data/models/model4.h5", 0.9, 0.1)


# encode_universal_adversarial_robustness("../../data/models/model0.h5", 0.15, 0.1)
# encode_universal_adversarial_robustness("../../data/models/model4.h5", 0.3, 0.6)
# encode_universal_adversarial_robustness("../../data/models/model5.h5", 0.2, 0.4)
# encode_universal_adversarial_robustness("../../data/models/model6.h5", 0.2, 0.4)
# encode_universal_adversarial_robustness("../../data/models/model7.h5", 0.4, 0.4)
# encode_universal_adversarial_robustness("../../data/models/model9.h5", 0.4, 0.4)
# encode_universal_adversarial_robustness("../../data/models/model9.h5", 0.1, 0.2)
