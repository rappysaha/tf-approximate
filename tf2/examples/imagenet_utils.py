"""
ImageNet Data Loading and Preprocessing Utilities for MobileNet-v2

This module provides functions to load and preprocess ImageNet data for training
and evaluation of MobileNet-v2 models with approximate convolutions.

Expected directory structure:
  imagenet_path/
    train/
      class_0/
        image_1.JPEG
        image_2.JPEG
        ...
      class_1/
        ...
    val/
      class_0/
        image_1.JPEG
        ...
"""

import os
import tensorflow as tf
from pathlib import Path


# ImageNet normalization constants (standard mean/std)
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])


def _parse_image_file(filepath):
    """
    Parse and decode image file from disk.

    Args:
        filepath: Path to image file (string tensor)

    Returns:
        Decoded image tensor (3D: height x width x channels)
    """
    raw_image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(raw_image, channels=3)
    return image


def _preprocess_image(image, training=False, augment=False):
    """
    Preprocess image for MobileNet-v2 (224x224 input).

    Args:
        image: Input image tensor (any size, 3 channels)
        training: Boolean, whether in training mode
        augment: Boolean, whether to apply augmentation (only for training)

    Returns:
        Preprocessed image tensor (224, 224, 3) normalized with ImageNet stats
    """
    # Resize to 256x256 (larger than target to allow cropping)
    image = tf.image.resize(image, (256, 256), method='bilinear')
    image = tf.cast(image, tf.float32)

    if augment and training:
        # Random crop 224x224 from 256x256
        image = tf.image.random_crop(image, size=[224, 224, 3])

        # Random horizontal flip (50% chance)
        image = tf.image.random_flip_left_right(image)

        # Random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)

        # Random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    else:
        # Center crop 224x224 (for validation/test without augmentation)
        offset_height = (256 - 224) // 2
        offset_width = (256 - 224) // 2
        image = tf.image.crop_to_bounding_box(
            image, offset_height, offset_width, 224, 224)

    # Normalize to [0, 1]
    image = image / 255.0

    # Apply ImageNet normalization (mean subtraction and std division)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image


def _load_image_and_label(filepath, label):
    """
    Load, decode, and preprocess image with its class label.

    Args:
        filepath: Path to image file (string tensor)
        label: Class label (integer tensor)

    Returns:
        Tuple of (preprocessed_image, label)
    """
    image = _parse_image_file(filepath)
    image = _preprocess_image(image, training=False, augment=False)
    return image, label


def load_imagenet_data(data_path, split='train', batch_size=128,
                       augment=False, shuffle=True, prefetch=True):
    """
    Load ImageNet dataset from directory structure.

    Expected structure:
      data_path/train/{class_id}/*.JPEG
      data_path/val/{class_id}/*.JPEG

    Args:
        data_path: Path to ImageNet root directory
        split: 'train', 'val', or custom split name
        batch_size: Batch size for data loading
        augment: Whether to apply data augmentation (only for training)
        shuffle: Whether to shuffle the dataset
        prefetch: Whether to enable prefetching for performance

    Returns:
        tf.data.Dataset with (images, labels) pairs
        - images: (batch_size, 224, 224, 3) float32 tensors
        - labels: (batch_size,) int32 class indices

    Raises:
        ValueError: If data_path doesn't exist or split directory is missing
        FileNotFoundError: If no images found in split directory
    """
    # Validate data_path
    data_path = Path(data_path)
    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")

    split_path = data_path / split
    if not split_path.exists():
        raise ValueError(f"Split path does not exist: {split_path}")

    # Get class directories
    class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in {split_path}")

    num_classes = len(class_dirs)

    # Collect all image paths and labels
    image_paths = []
    labels = []

    for class_id, class_dir in enumerate(class_dirs):
        image_files = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg')) + \
            list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.JPG'))

        if not image_files:
            print(f"Warning: No images found in {class_dir}")
            continue

        for img_path in image_files:
            image_paths.append(str(img_path))
            labels.append(class_id)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {split_path}")

    print(f"Loaded {len(image_paths)} images from {len(class_dirs)} classes")

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(
            image_paths), reshuffle_each_iteration=True)

    # Load and preprocess images
    # Use map with num_parallel_calls for efficient I/O
    dataset = dataset.map(
        lambda path, label: _load_image_and_label_with_aug(
            path, label, augment, split),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch
    dataset = dataset.batch(batch_size)

    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_classes


def _load_image_and_label_with_aug(filepath, label, augment, split):
    """
    Helper function to load, decode, and preprocess image with augmentation control.

    Args:
        filepath: Path to image file (string tensor)
        label: Class label (integer tensor)
        augment: Boolean, whether to apply augmentation
        split: Split name ('train' means training mode for augmentation)

    Returns:
        Tuple of (preprocessed_image, label)
    """
    image = _parse_image_file(filepath)
    training = (split == 'train')
    image = _preprocess_image(image, training=training, augment=augment)
    return image, label


def get_imagenet_dataset_info(data_path):
    """
    Get information about the ImageNet dataset structure.

    Args:
        data_path: Path to ImageNet root directory

    Returns:
        Dictionary with dataset statistics
    """
    data_path = Path(data_path)
    info = {}

    for split in ['train', 'val']:
        split_path = data_path / split
        if split_path.exists():
            class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
            total_images = sum(
                len(list(d.glob('*.JPEG')) + list(d.glob('*.jpg')) +
                    list(d.glob('*.jpeg')) + list(d.glob('*.JPG')))
                for d in class_dirs
            )
            info[split] = {
                'num_classes': len(class_dirs),
                'num_images': total_images
            }

    return info


# Example usage:
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test ImageNet data loading')
    parser.add_argument('--data_path', required=True,
                        help='Path to ImageNet root directory')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size')
    args = parser.parse_args()

    # Get dataset info
    info = get_imagenet_dataset_info(args.data_path)
    print(f"Dataset info: {info}")

    # Load a small batch for testing
    try:
        train_dataset, num_classes = load_imagenet_data(
            args.data_path, split='train', batch_size=args.batch_size, augment=True)

        print(f"Loaded training dataset with {num_classes} classes")

        # Get one batch
        for images, labels in train_dataset.take(1):
            print(f"Batch shape: images={images.shape}, labels={labels.shape}")
            print(f"Image value range: min={tf.reduce_min(images).numpy():.3f}, "
                  f"max={tf.reduce_max(images).numpy():.3f}")
            print(f"Labels: {labels.numpy()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
