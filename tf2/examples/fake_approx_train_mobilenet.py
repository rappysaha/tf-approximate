#!/usr/bin/env python3
"""
Train MobileNet-v2 on ImageNet for use with approximate convolutions.

This script fine-tunes a pretrained MobileNet-v2 model on ImageNet data,
saving the weights for later evaluation with FakeApproxConv2D layers.

Usage:
    python fake_approx_train_mobilenet.py \
        --imagenet_path /path/to/imagenet \
        --epochs 2 \
        --batch_size 128 \
        --learning_rate 0.001 \
        --save_dir models/

Expected ImageNet directory structure:
    imagenet_path/
        train/{class_id}/*.JPEG
        val/{class_id}/*.JPEG
"""

# python examples/fake_approx_train_mobilenet.py --imagenet_path /mnt/new_ssd/data_gabriel/tiny-imagenet-1000 --mtab_file examples/axmul_8x8/mul8u_L40.bin --epochs 2 --batch_size 8

# python examples/fake_approx_train_mobilenet.py --imagenet_path /mnt/new_ssd/data_gabriel/tiny-imagenet-1000 --mtab_file tf-approximate/tf2/examples/axmul_8x8/fame/SPRIM8_41.bin --epochs 2 --batch_size 8

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
        '--epochs',
        type=int,
        default=2,
        help='Number of training epochs (default: 2)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for training (default: 128)'
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
    parser.add_argument(
        '--mtab_file',
        default='',
        help='Approximate multiplication table for FakeApproxConv2D (default: none)'
    )
    parser.add_argument(
        '--steps_per_epoch',
        type=int,
        default=None,
        help='Limit training steps per epoch (for testing, default: None = use all)'
    )
    parser.add_argument(
        '--val_split_name',
        default='val',
        help='Name of validation split in ImageNet directory (default: val)'
    )

    args = parser.parse_args()

    # Validate inputs
    imagenet_path = Path(args.imagenet_path)
    if not imagenet_path.exists():
        print(f"Error: ImageNet path does not exist: {imagenet_path}")
        sys.exit(1)

    train_path = imagenet_path / 'train'
    val_path = imagenet_path / args.val_split_name

    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        sys.exit(1)

    if not val_path.exists():
        print(f"Error: Validation data not found at {val_path}")
        sys.exit(1)

    # Setup
    print("="*70)
    print("MobileNet-v2 Training on ImageNet")
    print("="*70)
    setup_gpu()

    # Load data
    print(f"\nLoading training data from {train_path}...")
    train_dataset, num_classes = load_imagenet_data(
        str(imagenet_path),
        split='train',
        batch_size=args.batch_size,
        augment=True,
        shuffle=True
    )

    print(f"Loading validation data from {val_path}...")
    val_dataset, _ = load_imagenet_data(
        str(imagenet_path),
        split=args.val_split_name,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False
    )

    # Build and compile model
    print("\nBuilding MobileNet-v2 model...")
    model = build_model(num_classes=num_classes)
    compile_model(model, learning_rate=args.learning_rate)

    # print("\nModel summary:")
    # model.summary()

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

    # Create callbacks
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    log_dir = os.path.join('tflogs', timestamp)

    # Train the model
    callbacks = create_callbacks(
        log_dir, args.save_dir, model_name='mobilenet_v2_exactFP32')
    print("\nStarting training with the accurate model...")
    history = train(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        steps_per_epoch=args.steps_per_epoch
    )
    # Save the final accurate model with weights
    exactFP32_model_save_path = Path(
        args.save_dir) / 'mobilenet_v2_exactFP32.keras'
    print(f"\nSaving final accurate model to {exactFP32_model_save_path}...")
    model.save(exactFP32_model_save_path)

    # Create callbacks for the approximate model
    approx_callbacks = create_callbacks(
        log_dir, args.save_dir, model_name='mobilenet_v2_approx')
    print("\nStarting training with the approximate model...")
    history_approx = train(
        approx_model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        callbacks=approx_callbacks,
        steps_per_epoch=args.steps_per_epoch
    )

    # Save the final approximate model with weights
    approx_model_save_path = Path(args.save_dir) / 'mobilenet_v2_approx.keras'
    print(f"\nSaving final approximate model to {approx_model_save_path}...")
    approx_model.save(approx_model_save_path)

    # Final evaluation on validation set  by loading the best checkpoints saved during training
    print("\nEvaluating final Exact model on validation set...")
    best_exact_model_path = Path(args.save_dir) / \
        'mobilenet_v2_exactFP32_best.keras'
    if best_exact_model_path.exists():
        print(f"Loading best Exact model from {best_exact_model_path}...")
        best_exact_model = tf.keras.models.load_model(best_exact_model_path)
        exact_eval = best_exact_model.evaluate(val_dataset, verbose=1)
        print(
            f"Best Exact Model - Val Loss: {exact_eval[0]:.4f}, Val Top-1 Accuracy: {exact_eval[1]:.4f}, Val Top-5 Accuracy: {exact_eval[2]:.4f}")
    else:
        print(
            f"Warning: Best Exact model checkpoint not found at {best_exact_model_path}")

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


def train(
    model,
    train_dataset,
    val_dataset,
    epochs=2,
    callbacks=None,
    steps_per_epoch=None
):
    """
    Train the model.

    Args:
        model: Compiled Keras model
        train_dataset: tf.data.Dataset for training
        val_dataset: tf.data.Dataset for validation
        epochs: Number of epochs to train
        callbacks: List of Keras callbacks
        steps_per_epoch: Optional limit on steps per epoch (for testing)

    Returns:
        Training history
    """
    print(f"\nStarting training for {epochs} epochs...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )

    return history


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


def create_callbacks(log_dir, checkpoint_dir, model_name='mobilenet_v2'):
    """
    Create training callbacks for TensorBoard logging and checkpointing.

    Args:
        log_dir: Directory for TensorBoard logs
        checkpoint_dir: Directory to save model checkpoints
        model_name: Name prefix for saved models

    Returns:
        List of Keras callbacks
    """
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard callback
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        update_freq='epoch',
        profile_batch=0
    )

    # Model checkpoint callback (save best model)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.keras')
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_top_1_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    )

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_top_1_accuracy',
        mode='max',
        patience=3,
        verbose=1,
        restore_best_weights=True
    )

    return [tensorboard_callback, checkpoint_callback, early_stop_callback]


if __name__ == '__main__':
    main()
