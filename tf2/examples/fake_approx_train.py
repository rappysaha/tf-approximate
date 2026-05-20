## ========== Copyright (c) 2020, Filip Vaverka, All rights reserved. =========##
##
# Purpose:     Train LeNet-5 on MNIST dataset.
##
# $NoKeywords: $ApproxTF $fake_approx_train.py
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

# python examples/fake_approx_train.py --mtab_file examples/axmul_8x8/mul8u_L40.bin --epochs 2

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mtab_file', type=str,
                    help='Approximate multiplication table (8x8)', default='')
parser.add_argument('--epochs', type=int,
                    help='Number of training epochs', default=6)

args = parser.parse_args()

# cuDNN can sometimes fail to initialize when TF reserves all of the GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# ============================================================================
# Load and prepare the MNIST dataset.
# ============================================================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# ============================================================================
# Define the LeNet-5 model architecture (exact multiplication)
# ============================================================================
print('='*80)
print('Defining Exact LeNet-5 model (standard Conv2D)')
print('='*80)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(
        3, 3), activation='relu', name='conv2d_1'),
    tf.keras.layers.AveragePooling2D(name='pool_1'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(
        3, 3), activation='relu', name='conv2d_2'),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(120, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(84, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(10, activation='softmax', name='output')
], name='lenet5_exact')

# Compile the exact model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("\nExact model summary:")
model.build(input_shape=(None, 28, 28, 1))
model.summary()

# ============================================================================
# Define the LeNet-5 model architecture (approximate multiplication)
# ============================================================================
print()
print('='*80)
print('Defining Approximate LeNet-5 model (FakeApproxConv2D)')
print('='*80)

approx_model = tf.keras.Sequential([
    FakeApproxConv2D(filters=6, kernel_size=(3, 3), activation='relu',
                     mul_map_file=args.mtab_file, name='approx_conv2d_1'),
    tf.keras.layers.AveragePooling2D(name='pool_1'),
    FakeApproxConv2D(filters=16, kernel_size=(3, 3), activation='relu',
                     mul_map_file=args.mtab_file, name='approx_conv2d_2'),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(120, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(84, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(10, activation='softmax', name='output')
], name='lenet5_approximate')

# Compile the approximate model
approx_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
print("\nApproximate model summary:")
approx_model.build(input_shape=(None, 28, 28, 1))
approx_model.summary()
# Save the approximate model architecture and weights
approx_model.save('models/lenet5_approx_model.h5')

# ============================================================================
# Train both models
# ============================================================================
print()
print('='*80)
print('Training Exact Model')
print('='*80)

tensorboard_dir = "tflogs/{}".format(
    datetime.datetime.now().replace(microsecond=0).isoformat())
tensorboard = TensorBoard(log_dir=tensorboard_dir)

history_exact = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=args.epochs,
    callbacks=[tensorboard],
    verbose=1
)

print()
print('='*80)
print('Training Approximate Model with {}'.format(args.mtab_file))
print('='*80)

history_approx = approx_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=args.epochs,
    callbacks=[tensorboard],
    verbose=1
)

# ============================================================================
# Evaluate both models on test set
# ============================================================================
print()
print('='*80)
print('Evaluating Exact Model on Test Set')
print('='*80)

exact_score = model.evaluate(x_test, y_test, verbose=0)
print('Exact Model - Test Loss: {:.6f}, Test Accuracy: {:.6f}'.format(
    exact_score[0], exact_score[1]))

print()
print('='*80)
print('Evaluating Approximate Model on Test Set')
print('='*80)

approx_score = approx_model.evaluate(x_test, y_test, verbose=0)
print('Approximate Model - Test Loss: {:.6f}, Test Accuracy: {:.6f}'.format(
    approx_score[0], approx_score[1]))

# ============================================================================
# Compare the results
# ============================================================================
print()
print('='*80)
print('COMPARISON SUMMARY')
print('='*80)
print('Model Type              | Loss         | Accuracy')
print('-'*80)
print('Exact (Conv2D)          | {:.6f}   | {:.6f}'.format(
    exact_score[0], exact_score[1]))
print('Approximate (FakeApprox)| {:.6f}   | {:.6f}'.format(
    approx_score[0], approx_score[1]))
print('-'*80)
loss_diff = abs(exact_score[0] - approx_score[0])
acc_diff = abs(exact_score[1] - approx_score[1])
print('Loss Difference         | {:.6f}'.format(loss_diff))
print('Accuracy Difference     | {:.6f}'.format(acc_diff))
print('='*80)

# ============================================================================
# Save both models (both weights and full models)
# ============================================================================
print()
print('='*80)
print('Saving Models')
print('='*80)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save exact model
model.save_weights('models/lenet5_weights')
model.save('models/lenet5_exact_model.keras')
print('✓ Exact model weights saved to: models/lenet5_weights')
print('✓ Exact model (full) saved to: models/lenet5_exact_model.keras')

# Save approximate model
# NOTE: Due to non-serializable custom ops in the FakeApproxConv2D layer,
#       we save weights only (RestrictedSerializer would fail). Weights are sufficient for inference.
approx_model.save_weights('models/lenet5_approx_weights')
print('✓ Approximate model weights saved to: models/lenet5_approx_weights')
print('  (Note: Full model saved via weights only due to custom op serialization limitations)')

print()
print('='*80)
print('Training Complete!')
