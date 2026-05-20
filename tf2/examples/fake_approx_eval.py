## ========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
# Purpose:     Evaluate LeNet-5 with approximate Conv2D layers.
##
# $NoKeywords: $ApproxTF $fake_approx_eval.py
# $Date:       $2020-02-25
## ============================================================================##

import sys
import os
import importlib.util
import datetime
import argparse
import tensorflow as tf
# Use attribute access on the imported tensorflow module to obtain TensorBoard
# instead of `from tensorflow.keras...` which triggers a submodule import that
# can conflict with a local `keras` package on PYTHONPATH.
TensorBoard = tf.keras.callbacks.TensorBoard


# Load local FakeApproxConv2D module directly to avoid import conflicts
_this_dir = os.path.dirname(os.path.abspath(__file__))
_local_layer_path = os.path.abspath(os.path.join(
    _this_dir, '..', 'python', 'keras', 'layers', 'fake_approx_convolutional.py'))
spec = importlib.util.spec_from_file_location(
    "fake_approx_convolutional", _local_layer_path)
fake_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake_mod)
FakeApproxConv2D = fake_mod.FakeApproxConv2D

# cuDNN can sometimes fail to initialize when TF reserves all of the GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mtab_file', type=str,
                    help='Approximate multiplication table (8x8)', default='')

args = parser.parse_args()

# Load and prepare the MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# print(x_train.shape)

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

# print(x_train.shape)

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Define our approximate model architecture
# NOTE: Conv2D layers are replaced with our FakeApproxConv2D which simulates convolutional layer with approximate
#       8bit fixed-point multiplication.
approx_model = tf.keras.Sequential([
    FakeApproxConv2D(filters=6, kernel_size=(
        3, 3), activation='relu', mul_map_file=args.mtab_file),
    tf.keras.layers.AveragePooling2D(),
    FakeApproxConv2D(filters=16, kernel_size=(
        3, 3), activation='relu', mul_map_file=args.mtab_file),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
approx_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
approx_model.build(input_shape=(0, 28, 28, 1))

approx_model.load_weights('models/lenet5_approx_weights')

# NOTE: Weights can also be directly copied from the trained model (instead of loading stored ones)
# for approx_layer, layer in zip(approx_model.layers, model.layers):
#     approx_layer.set_weights(layer.get_weights())

# print('================================================================================')
# print('Training approximate model with {}'.format(args.mtab_file))
# # Connect to Tensorboard and train the model
# tensorboard = TensorBoard(
#     log_dir="tflogs/{}".format(datetime.datetime.now().replace(microsecond=0).isoformat()))

# approx_model.fit(x_train, y_train, validation_data=(
#     x_test, y_test), epochs=6, callbacks=[tensorboard])

# approx_model.save_weights('models/lenet5_approx_weights')

# approx_model.load_weights('models/lenet5_approx_weights')

print('================================================================================')
print('Testing approximate model with {}'.format(args.mtab_file))
score = approx_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
