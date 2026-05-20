#!/usr/bin/env python3
"""
Evaluate MobileNet-v2 with Approximate Convolutions on ImageNet.

This script loads a trained MobileNet-v2 model and evaluates it in two forms:
1. Exact FP32: Standard Conv2D layers (baseline)
2. Approximate model: Conv2D replaced with FakeApproxConv2D (using lookup tables)

Compares performance metrics and reports degradation due to approximation.

Usage:
    python examples/fake_approx_eval_mobilenet.py \
        --imagenet_path /mnt/new_ssd/data_gabriel/tiny-imagenet-1000 \
        --mtab_file examples/axmul_8x8/mul8u_1JFF.bin \
        --batch_size 16
    python examples/fake_approx_eval_mobilenet.py \
        --imagenet_path /mnt/new_ssd/data_gabriel/ILSVRC/Data/CLS-LOC \
        --mtab_file examples/axmul_8x8/mul8u_L40.bin \
        --batch_size 16

Expected ImageNet directory structure:
    imagenet_path/
        val/{class_id}/*.JPEG
"""

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from imagenet_utils import load_imagenet_data
from approx_model_builder import replace_conv2d_with_approx, compare_model_layers
from tf_cuda_setup import configure_cuda_runtime
import os
import sys
import argparse
import datetime
import importlib.util
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


configure_cuda_runtime()


def setup_gpu():
    """Configure GPU memory growth to avoid cuDNN initialization failures."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(
                f"GPU memory growth enabled for {len(physical_devices)} device(s)")
        except RuntimeError as e:
            print(f"Warning: Could not enable GPU memory growth: {e}")
    else:
        print("No GPU devices found. Training will use CPU (may be slow).")


def main():
    parser = argparse.ArgumentParser(
        description='Train MobileNet-v2 on ImageNet for approximate evaluation'
    )
    parser.add_argument(
        '--imagenet_path',
        required=True,
        help='Path to ImageNet dataset root (expects train/ and val/ subdirectories)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    parser.add_argument(
        '--mtab_file',
        default='',
        help='Approximate multiplication table for FakeApproxConv2D (default: none)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for Adam optimizer (default: 0.001)'
    )
    parser.add_argument(
        '--save_dir',
        default='models',
        help='Directory to save trained weights (default: models)'
    )

    args = parser.parse_args()

    # Validate inputs
    imagenet_path = Path(args.imagenet_path)
    if not imagenet_path.exists():
        print(f"Error: ImageNet path does not exist: {imagenet_path}")
        sys.exit(1)

    val_path = imagenet_path / 'val'

    if not val_path.exists():
        print(f"Error: Validation data not found at {val_path}")
        sys.exit(1)

    # Setup
    print("="*70)
    print("MobileNet-v2 Evaluation on ImageNet")
    print("="*70)
    setup_gpu()

    # Load data
    print(f"\nLoading validation data from {val_path}...")
    val_dataset, num_classes = load_imagenet_data(
        str(imagenet_path),
        split='val',
        batch_size=args.batch_size,
        augment=False,
        shuffle=False
    )

    # Build and compile model
    print("\nBuilding MobileNet-v2 model...")
    model = build_model(num_classes=num_classes)
    compile_model(model, learning_rate=args.learning_rate)

    # Build approximate model
    print("\nBuilding approximate model with FakeApproxConv2D layers...")
    approx_model = None
    try:
        approx_model = replace_conv2d_with_approx(
            model,
            mul_map_file=args.mtab_file if args.mtab_file else None,
            verbose=False
        )
        compile_model(approx_model, learning_rate=args.learning_rate)
    except Exception as e:
        print(f"Warning: Could not build approximate model: {e}")
        print("Continuing with accurate-model-only evaluation.")
        import traceback
        traceback.print_exc()
        approx_model = None
    compare_model_layers(model, approx_model, verbose=False)

    exact_eval = model.evaluate(val_dataset, verbose=1)
    print(
        f"Best Exact Model - Val Loss: {exact_eval[0]:.4f}, Val Top-1 Accuracy: {exact_eval[1]:.4f}, Val Top-5 Accuracy: {exact_eval[2]:.4f}")

    print("\nEvaluating final Approximate model on validation set...")
    best_approx_model_path = Path(
        args.save_dir) / 'mobilenet_v2_approx_best.keras'
    if best_approx_model_path.exists():
        print(
            f"Loading best Approximate model from {best_approx_model_path}...")
        # Load weights into the current approx_model architecture
        approx_model.load_weights(best_approx_model_path)
        # best_approx_model = tf.keras.models.load_model(best_approx_model_path)## does not work
        approx_eval = approx_model.evaluate(val_dataset, verbose=1)
        print(
            f"Best Approximate Model - Val Loss: {approx_eval[0]:.4f}, Val Top-1 Accuracy: {approx_eval[1]:.4f}, Val Top-5 Accuracy: {approx_eval[2]:.4f}")
    else:
        print(
            f"Warning: Best Approximate model checkpoint not found at {best_approx_model_path}")


def build_model(num_classes=1000):
    """
    Build MobileNet-v2 model with ImageNet pretrained weights.

    Args:
        num_classes: Number of output classes (1000 for full ImageNet)

    Returns:
        Compiled Keras model ready for training
    """
    print("Loading MobileNet-v2 with ImageNet pretrained weights...")

    # If num_classes == 1000 we can load the full pretrained model including top
    if num_classes == 1000:
        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights='imagenet',
            classes=1000
        )
        return model

    # Otherwise load base (no top) and attach a new classification head
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    x = base.output
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(
        inputs=base.input, outputs=outputs, name='mobilenet_v2_custom')

    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate loss, optimizer, and metrics.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
    """
    print(f"Compiling model with learning rate {learning_rate}...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=1, name='top_1_accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name='top_5_accuracy')
        ]
    )


if __name__ == '__main__':
    main()
