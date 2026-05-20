"""
Approximate Model Builder - Convert Conv2D layers to FakeApproxConv2D

This module provides utilities to transform standard Keras models by replacing
Conv2D layers with FakeApproxConv2D layers for approximate computation evaluation.

The conversion maintains layer weights and configuration while enabling the use of
approximate multiplier lookup tables during forward passes.
"""

import tensorflow as tf
from tf_cuda_setup import configure_cuda_runtime
import os
import sys
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


configure_cuda_runtime()

models = tf.keras.models
Conv2D = tf.keras.layers.Conv2D
DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
Conv2DTranspose = tf.keras.layers.Conv2DTranspose


def _load_fake_approx_conv2d_layer():
    """
    Dynamically load FakeApproxConv2D layer from local module.

    Avoids import conflicts by loading the layer directly from the source file.

    Returns:
        FakeApproxConv2D class

    Raises:
        ImportError: If FakeApproxConv2D module cannot be found
    """
    # Get the path relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)  # tf2 root
    layer_path = os.path.abspath(os.path.join(
        parent_dir, 'python', 'keras', 'layers', 'fake_approx_convolutional.py'))

    if not os.path.exists(layer_path):
        raise ImportError(f"Cannot find FakeApproxConv2D at {layer_path}")

    # Load module dynamically
    spec = importlib.util.spec_from_file_location(
        "fake_approx_convolutional", layer_path)
    fake_mod = importlib.util.module_from_spec(spec)
    sys.modules['fake_approx_convolutional'] = fake_mod
    spec.loader.exec_module(fake_mod)

    return fake_mod.FakeApproxConv2D


def replace_conv2d_with_approx(model, mul_map_file=None, verbose=True):
    """
    Replace Conv2D layers in a Keras model with FakeApproxConv2D layers.
    """
    FakeApproxConv2D = _load_fake_approx_conv2d_layer()

    replacements = {'total': 0, 'replaced': 0, 'skipped': 0}

    if verbose:
        print(f"Converting model: {model.name}")
        print(f"Model type: {type(model).__name__}")
        print(f"\nPhase 1: Analyzing layers...")

    for orig_layer in model.layers:
        is_standard_conv2d = isinstance(orig_layer, Conv2D) and not isinstance(
            orig_layer, (DepthwiseConv2D, Conv2DTranspose))

        if is_standard_conv2d:
            replacements['total'] += 1
            replacements['replaced'] += 1
            if verbose:
                print(f"  → Will replace Conv2D '{orig_layer.name}'")
        elif isinstance(orig_layer, Conv2D):
            replacements['total'] += 1
            replacements['skipped'] += 1
            if verbose:
                print(
                    f"  ○ Skip {type(orig_layer).__name__} '{orig_layer.name}'")

    if verbose:
        print(f"\nPhase 2: Rebuilding model from edited config...")

    model_config = model.get_config()
    for layer_cfg in model_config.get('layers', []):
        if layer_cfg.get('class_name') == 'Conv2D':
            layer_cfg['class_name'] = 'FakeApproxConv2D'
            layer_cfg['module'] = None
            layer_cfg['registered_name'] = 'ApproxTF>FakeApproxConv2D'
            layer_cfg['config']['mul_map_file'] = mul_map_file or ''
            layer_cfg['config']['num_bits'] = layer_cfg['config'].get(
                'num_bits', 8)

    try:
        approx_model = tf.keras.Model.from_config(
            model_config,
            custom_objects={'FakeApproxConv2D': FakeApproxConv2D}
        )
    except Exception as e:
        raise ValueError(f"Failed to rebuild model: {e}")

    fake_approx_layers = [
        layer for layer in approx_model.layers if isinstance(layer, FakeApproxConv2D)
    ]

    if verbose:
        print(f"\nPhase 3: Checking rebuilt layer types before weight transfer...")
        print(f"  Model has {len(model.layers)} original layers")
        print(f"  Rebuilt model has {len(approx_model.layers)} layers")
        print(
            f"  FakeApproxConv2D instances visible in approx_model: {len(fake_approx_layers)}")
        for index, layer in enumerate(fake_approx_layers[:5], 1):
            print(f"    [{index}] {layer.name} ({type(layer).__name__})")

    if verbose:
        print(f"\nPhase 4: Transferring weights...")

    original_layers_by_name = {layer.name: layer for layer in model.layers}
    approx_layers_by_name = {
        layer.name: layer for layer in approx_model.layers}
    weight_transfer_count = 0

    for layer_name, orig_layer in original_layers_by_name.items():
        cloned_layer = approx_layers_by_name.get(layer_name)
        if cloned_layer is None:
            if verbose:
                print(f"  ⚠ Missing cloned layer for '{layer_name}'")
            continue

        try:
            orig_weights = orig_layer.get_weights()
            if orig_weights:
                cloned_layer.set_weights(orig_weights)
                weight_transfer_count += 1
                if verbose and isinstance(cloned_layer, FakeApproxConv2D):
                    print(f"  ✓ Transferred weights for '{cloned_layer.name}'")
        except Exception as e:
            if verbose and isinstance(cloned_layer, FakeApproxConv2D):
                print(
                    f"  ⚠ Could not transfer weights for '{cloned_layer.name}': {e}")

    if verbose:
        print(f"\n{'='*70}")
        print(f"Conversion Summary:")
        print(f"  Total Conv-like layers: {replacements['total']}")
        print(f"  Successfully replaced: {replacements['replaced']}")
        print(f"  Skipped (special variants): {replacements['skipped']}")
        print(f"  Weights transferred: {weight_transfer_count}")

    if verbose:
        print(f"\n{'='*70}")
        print(f"Verification: Checking rebuilt model layer types...")
        print(f"{'='*70}")
        print(
            f"Found {len(fake_approx_layers)} FakeApproxConv2D layers in approx_model")
        for i, layer in enumerate(fake_approx_layers, 1):
            print(f"\n  [{i}] FakeApproxConv2D '{layer.name}':")
            print(f"      - Type: {type(layer).__name__}")
            print(
                f"      - isinstance check: {isinstance(layer, FakeApproxConv2D)}")
            print(f"      - Filters: {layer.filters}")
            print(f"      - Kernel size: {layer.kernel_size}")
            print(f"      - Strides: {layer.strides}")
            print(f"      - Padding: {layer.padding}")
            print(f"      - Dilation rate: {layer.dilation_rate}")
            print(f"      - Use bias: {layer.use_bias}")
            print(f"      - Mul map file: {layer.mul_map_file}")
            weights = layer.get_weights()
            if weights:
                print(f"      - Weights: {len(weights)} arrays loaded")
                for j, w in enumerate(weights):
                    print(
                        f"        • Array {j}: shape {w.shape}, dtype {w.dtype}")
            else:
                print(f"      - Weights: NOT loaded")
        print(f"\n{'='*70}\n")

    if len(fake_approx_layers) != replacements['replaced']:
        print(
            f"⚠ Warning: Expected {replacements['replaced']} FakeApproxConv2D layers but found {len(fake_approx_layers)} in approx_model")

    return approx_model


def _replace_layer_if_needed(layer, FakeApproxConv2D, layer_mapping, mul_map_file, replacements, verbose):
    """Helper function to replace Conv2D layers during cloning."""
    if isinstance(layer, Conv2D) and not isinstance(layer, (DepthwiseConv2D, Conv2DTranspose)):
        if layer.name in layer_mapping:
            try:
                config = layer.get_config()
                config['mul_map_file'] = mul_map_file or ''
                new_layer = FakeApproxConv2D.from_config(config)
                replacements['replaced'] += 1
                if verbose:
                    print(f"  ✓ Created FakeApproxConv2D '{layer.name}'")
                return new_layer
            except Exception as e:
                replacements['failed'] += 1
                if verbose:
                    print(
                        f"  ✗ Failed to create FakeApproxConv2D '{layer.name}': {e}")
    return layer


def build_approximate_mobilenet_v2(weights_path=None, mul_map_file=None,
                                   include_top=True, num_classes=1000):
    """
    Build a MobileNet-v2 model with FakeApproxConv2D layers.

    Creates a MobileNet-v2 model from tf.keras.applications with ImageNet weights,
    then converts all Conv2D layers to FakeApproxConv2D for approximate evaluation.

    Args:
        weights_path: Optional path to custom weights (.h5 or SavedModel directory)
                     If None, uses ImageNet pretrained weights
        mul_map_file: Path to approximate multiplier lookup table
                     If None or '', uses accurate 8x8 multiplication
        include_top: Whether to include classification top (1000-class head)
        num_classes: Number of classes (for custom head if include_top=True)

    Returns:
        Keras model (MobileNet-v2 with FakeApproxConv2D layers)

    Raises:
        FileNotFoundError: If weights_path doesn't exist
        ValueError: If model conversion fails
    """
    print("Building MobileNet-v2 with approximate convolutions...")

    # Load base MobileNet-v2
    if weights_path is None:
        print("  Loading MobileNet-v2 with ImageNet pretrained weights...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=include_top,
            weights='imagenet',
            classes=num_classes
        )
    else:
        print(f"  Loading MobileNet-v2 from {weights_path}...")
        if os.path.isdir(weights_path):
            # SavedModel format
            base_model = tf.keras.models.load_model(weights_path)
        else:
            # HDF5 format
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=include_top,
                weights=None,
                classes=num_classes
            )
            base_model.load_weights(weights_path)

    # Convert Conv2D to FakeApproxConv2D
    approx_model = replace_conv2d_with_approx(
        base_model, mul_map_file=mul_map_file)

    return approx_model


def compare_model_layers(original_model, approx_model, verbose=True):
    """
    Compare original and approximate models to verify conversion.

    Args:
        original_model: Original Keras model
        approx_model: Approximate Keras model
        verbose: Print comparison details

    Returns:
        Dictionary with comparison statistics
    """
    stats = {
        'total_layers': len(original_model.layers),
        'conv2d_layers': 0,
        'approx_conv2d_layers': 0,
        'other_layers': 0,
        'weight_transfer_success': 0,
        'weight_transfer_failed': 0
    }

    if verbose:
        print("\nModel Layer Comparison:")
        print(
            f"{'Layer #':<6} {'Original Type':<25} {'Approximate Type':<25} {'Weights':<10}")
        print("-" * 70)

    for i, (orig_layer, approx_layer) in enumerate(zip(original_model.layers, approx_model.layers)):
        orig_type = type(orig_layer).__name__
        approx_type = type(approx_layer).__name__

        # Check weight transfer
        orig_weights = orig_layer.get_weights()
        approx_weights = approx_layer.get_weights()

        if orig_weights and approx_weights:
            # Compare weights
            if len(orig_weights) == len(approx_weights):
                match = all(
                    (w1 == w2).numpy().all() if isinstance(
                        w1, tf.Tensor) else (w1 == w2).all()
                    for w1, w2 in zip(orig_weights, approx_weights)
                )
                weights_status = "✓" if match else "✗"
                if match:
                    stats['weight_transfer_success'] += 1
                else:
                    stats['weight_transfer_failed'] += 1
            else:
                weights_status = "✗"
                stats['weight_transfer_failed'] += 1
        elif not orig_weights and not approx_weights:
            weights_status = "○"
        else:
            weights_status = "✗"
            stats['weight_transfer_failed'] += 1

        # Count layer types
        if 'Conv2D' in orig_type and not any(x in orig_type for x in ['Depthwise', 'Transpose']):
            stats['conv2d_layers'] += 1
        else:
            stats['other_layers'] += 1

        if 'FakeApproxConv2D' in approx_type:
            stats['approx_conv2d_layers'] += 1

        if verbose:
            print(f"{i:<6} {orig_type:<25} {approx_type:<25} {weights_status:<10}")

    if verbose:
        print("\nConversion Statistics:")
        print(f"  Total layers: {stats['total_layers']}")
        print(f"  Conv2D layers: {stats['conv2d_layers']}")
        print(
            f"  Converted to FakeApproxConv2D: {stats['approx_conv2d_layers']}")
        print(f"  Other layers: {stats['other_layers']}")
        print(
            f"  Weights transferred successfully: {stats['weight_transfer_success']}")
        print(f"  Weights transfer failed: {stats['weight_transfer_failed']}")

    return stats


# Example usage:
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test model conversion')
    parser.add_argument('--weights_path', default=None,
                        help='Path to trained weights')
    parser.add_argument('--mtab_file', default=None,
                        help='Path to multiplier table')
    args = parser.parse_args()

    try:
        # Build models
        print("Loading original MobileNet-v2...")
        original_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights='imagenet'
        )

        print("\nConverting to approximate model...")
        approx_model = replace_conv2d_with_approx(
            original_model, mul_map_file=args.mtab_file)

        print("\nComparing models...")
        stats = compare_model_layers(original_model, approx_model)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
