import random
import tensorflow as tf
import larq as lq
import matplotlib.pyplot as plt

from src.encodeModel.encode import Encode

# Load MNIST
_, (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images / 127.5 - 1  # Normalize pixel values to be between -1 and 1

# Print image
id_image = random.randint(0, 10000-1)
plt.imshow(test_images[id_image])
plt.show()

# Get stat
model = tf.keras.models.load_model("../data/models/model" + input("Enter the model number: ") + ".h5")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy {test_acc * 100:.2f} %")
lq.models.summary(model)

# Get intermediate values
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
acts = activation_model.predict(test_images[id_image:id_image+1])

print('\n- - - - - -', 'start', '- - - - - -')
for i in test_images[id_image]:
    for j in i:
        print(j, end='')
    print()

for i in range(len(model.layers)):
    layer = model.layers[i]
    print('\n-----------', layer.name, '-----------')
    print(layer.weights)
    print('\n- - - - - -', layer.name, '- - - - - -')
    print(acts[i].shape)
    print(acts[i])

print('\n-----------', 'ANSWER', '-----------')
max_value = max(acts[-1][0])
print([index for index, val in enumerate(acts[-1][0]) if val == max_value])

encode = Encode(model.layers)
encode.encode()
encode.print_vars()
encode.print_constraints()
