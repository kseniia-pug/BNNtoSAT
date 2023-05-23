import tensorflow as tf
from src.createModel.create_model import CreateModel0, CreateModel1, CreateModel2, CreateModel3, CreateModel4, \
    CreateModel5

# Load MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1 # Normalize pixel values to be between -1 and 1

CreateModel0(train_images, train_labels, test_images, test_labels)
# CreateModel1(train_images, train_labels, test_images, test_labels)
# CreateModel2(train_images, train_labels, test_images, test_labels)
# CreateModel3(train_images, train_labels, test_images, test_labels)
# CreateModel4(train_images, train_labels, test_images, test_labels)
# CreateModel5(train_images, train_labels, test_images, test_labels)
