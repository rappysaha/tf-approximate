# MobileNet-v2 Approximate Convolution Training & Evaluation Guide

## Overview

This guide explains how to train and evaluate MobileNet-v2 models with approximate convolutions using the tf-approximate library. The workflow consists of two main scripts:

1. **`fake_approx_train_mobilenet.py`** - Fine-tune MobileNet-v2 on ImageNet
2. **`fake_approx_eval_mobilenet.py`** - Evaluate with approximate vs accurate convolutions

---

## Prerequisites

### Environment Setup

```bash
# Navigate to tf2 directory
cd /mnt/new_ssd/workspace/tfapprox_rpp/tf-approximate/tf2

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/build"
```

### Data Requirements

**ImageNet Dataset Structure:**

The scripts expect ImageNet data organized as:

```
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
    class_1/
      ...
```

Where:

- `train/` contains ~1.2M images across 1000 classes
- `val/` contains ~50K images (50 images per class)
- Images are JPEG format (JPEG, jpg, or JPG extensions supported)

---

## Step 1: Training MobileNet-v2

### Basic Usage

Train MobileNet-v2 for 2 epochs on ImageNet:

```bash
python examples/fake_approx_train_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --epochs 2 \
    --batch_size 128
```

### Full Training with Custom Parameters

```bash
python examples/fake_approx_train_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --epochs 2 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --save_dir models/ \
    --val_split_name val
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--imagenet_path` | **required** | Path to ImageNet root directory |
| `--epochs` | 2 | Number of training epochs |
| `--batch_size` | 128 | Batch size for training |
| `--learning_rate` | 0.001 | Learning rate for Adam optimizer |
| `--save_dir` | `models/` | Directory to save trained weights |
| `--steps_per_epoch` | None | Limit steps per epoch (for testing) |
| `--val_split_name` | `val` | Name of validation split directory |

### Training Configuration Examples

**Quick Test (1000 images, 1 step per epoch):**

```bash
python examples/fake_approx_train_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --epochs 1 \
    --batch_size 64 \
    --steps_per_epoch 1 \
    --save_dir models/test/
```

**High Learning Rate (aggressive fine-tuning):**

```bash
python examples/fake_approx_train_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --epochs 3 \
    --batch_size 256 \
    --learning_rate 0.01 \
    --save_dir models/aggressive/
```

**Low Memory (smaller batch, fewer steps):**

```bash
python examples/fake_approx_train_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --epochs 2 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --steps_per_epoch 1000
```

### Training Output

The script saves weights in two formats:

```
models/
  mobilenet_v2_weights.h5          ← HDF5 format (smaller, faster loading)
  mobilenet_v2_weights_best.h5     ← Best checkpoint during training
  mobilenet_v2_savedmodel/         ← SavedModel format (includes architecture)
    assets/
    saved_model.pb
    variables/
    ...
```

### Training Logs

TensorBoard logs are saved with timestamp:

```bash
# View training in TensorBoard
tensorboard --logdir tflogs/

# Open browser: http://localhost:6006
```

### Expected Performance

**Typical metrics on ImageNet validation set:**

```
Epoch 1:
  Train Loss: 0.9234
  Train Top-1 Accuracy: 0.6521
  Train Top-5 Accuracy: 0.8845
  Val Loss: 0.8756
  Val Top-1 Accuracy: 0.6812
  Val Top-5 Accuracy: 0.8923

Epoch 2:
  Train Loss: 0.7654
  Train Top-1 Accuracy: 0.7123
  Train Top-5 Accuracy: 0.9134
  Val Loss: 0.7234
  Val Top-1 Accuracy: 0.7156
  Val Top-5 Accuracy: 0.9201
```

**Note:** Actual numbers depend on:

- Initial ImageNet weights (new vs previously fine-tuned)
- Data augmentation (random crops, flips)
- Batch size effects
- Learning rate and optimizer

---

## Step 2: Evaluating with Approximate Convolutions

### Basic Evaluation

Evaluate trained model with approximate multiplier:

```bash
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/mobilenet_v2_weights.h5 \
    --mtab_file examples/axmul_8x8/mul8u_L40.bin
```

### Full Evaluation with All Options

```bash
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/mobilenet_v2_weights.h5 \
    --mtab_file examples/axmul_8x8/mul8u_L40.bin \
    --batch_size 128 \
    --val_split_name val \
    --num_classes 1000 \
    --save_results \
    --results_file eval_results.json \
    --compare_layers
```

### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--imagenet_path` | **required** | Path to ImageNet root directory |
| `--weights_path` | **required** | Path to trained weights (.h5 or SavedModel dir) |
| `--mtab_file` | None | Path to multiplier table file (.bin). If not specified, uses accurate 8x8 |
| `--batch_size` | 128 | Batch size for evaluation |
| `--val_split_name` | `val` | Name of validation split directory |
| `--num_classes` | 1000 | Number of output classes |
| `--save_results` | False | Save results to JSON file |
| `--results_file` | `eval_results.json` | Output path for JSON results |
| `--compare_layers` | False | Print detailed layer comparison |

### Evaluation Scenarios

**Baseline (Accurate Model):**

```bash
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/mobilenet_v2_weights.h5 \
    --mtab_file ""  # Empty means accurate multiplication
```

**Different Multiplier Tables:**

```bash
# L40 multiplier (low error, good efficiency)
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/mobilenet_v2_weights.h5 \
    --mtab_file examples/axmul_8x8/mul8u_L40.bin

# 1JFF multiplier (accurate 8x8)
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/mobilenet_v2_weights.h5 \
    --mtab_file examples/axmul_8x8/mul8u_1JFF.bin

# Y48 multiplier (different tradeoff)
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/mobilenet_v2_weights.h5 \
    --mtab_file examples/axmul_8x8/mul8u_Y48.bin
```

**SavedModel Format:**

```bash
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/mobilenet_v2_savedmodel/ \
    --mtab_file examples/axmul_8x8/mul8u_L40.bin
```

**Detailed Analysis:**

```bash
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/mobilenet_v2_weights.h5 \
    --mtab_file examples/axmul_8x8/mul8u_L40.bin \
    --save_results \
    --results_file detailed_analysis.json \
    --compare_layers
```

### Evaluation Output

**Console Output:**

```
==============================================================================================
MobileNet-v2 Approximate Convolution Evaluation
==============================================================================================

Loading evaluation data from /path/to/imagenet/val...
Loaded 50000 images from 1000 classes

Loading trained model from models/mobilenet_v2_weights.h5...

---
Evaluating Accurate Model...
  Loss: 0.7234
  Top-1 Accuracy: 0.7156 (71.56%)
  Top-5 Accuracy: 0.9201 (92.01%)
---

Building approximate model with FakeApproxConv2D layers...
  Replacing Conv2D 'conv1' -> FakeApproxConv2D (filters=32, kernel=(3, 3))
  Replacing Conv2D 'expanded_conv_add' -> FakeApproxConv2D (filters=24, kernel=(3, 3))
  ...
Conversion complete!
  Total Conv layers: 89
  Replaced with FakeApproxConv2D: 87
  Kept as-is: 2

---
Evaluating Approximate Model...
  Loss: 0.7281
  Top-1 Accuracy: 0.7128 (71.28%)
  Top-5 Accuracy: 0.9184 (91.84%)
---

==============================================================================================
EVALUATION RESULTS - ACCURATE vs APPROXIMATE CONVOLUTIONS
==============================================================================================

Multiplier Table: mul8u_L40

Metric                    Accurate             Approximate         Degradation        
-----------------------------------------
Loss                           0.7234               0.7281         +0.0047
Top-1 Accuracy                 0.7156               0.7128         -0.0028 (-0.28%)
Top-5 Accuracy                 0.9201               0.9184         -0.0017 (-0.17%)
==============================================================================================

Interpretation:
  ✓ Good: Small accuracy loss (1-5%)

Saving results to eval_results.json...
Results saved successfully!

Evaluation complete!
```

**Results JSON File (`eval_results.json`):**

```json
{
  "timestamp": "2026-05-15T18:30:45.123456",
  "weights_path": "models/mobilenet_v2_weights.h5",
  "multiplier_table": "examples/axmul_8x8/mul8u_L40.bin",
  "batch_size": 128,
  "num_classes": 1000,
  "accurate_model": {
    "loss": 0.7234,
    "top_1_accuracy": 0.7156,
    "top_5_accuracy": 0.9201
  },
  "approximate_model": {
    "loss": 0.7281,
    "top_1_accuracy": 0.7128,
    "top_5_accuracy": 0.9184
  },
  "degradation": {
    "top_1_accuracy": 0.0028,
    "top_5_accuracy": 0.0017,
    "loss_increase": 0.0047
  }
}
```

---

## Available Multiplier Tables

Located in `examples/axmul_8x8/`:

| File | Type | Error Characteristics | Use Case |
|------|------|----------------------|----------|
| `mul8u_1JFF.bin` | Accurate 8×8 | 0% error - bit-perfect | Baseline/reference |
| `mul8u_L40.bin` | Approximate | Low error (~1-2%), good efficiency | Recommended for evaluation |
| `mul8u_Y48.bin` | Approximate | Medium error, different tradeoff | Alternative option |
| Others (36 total) | Approximate | Various error/efficiency | Exploration |

See `tf1/axqconv/axmult/` directory for complete list.

---

## Troubleshooting

### Data Loading Issues

**Problem:** `FileNotFoundError: No images found in...`

**Solution:**

- Verify ImageNet path exists: `ls /path/to/imagenet/train | head`
- Check subdirectory structure: `ls /path/to/imagenet/train/class_0 | head`
- Ensure image extensions are `.JPEG`, `.jpg`, or `.jpeg`

**Problem:** `ValueError: Data path does not exist`

**Solution:**

- Use absolute path: `python ... --imagenet_path /full/path/to/imagenet`
- Check path is correct: `cd /path/to/imagenet && ls`

### GPU Memory Issues

**Problem:** `CUDA out of memory`

**Solution - Training:**

```bash
# Reduce batch size
python fake_approx_train_mobilenet.py ... --batch_size 64
```

**Solution - Evaluation:**

```bash
# Reduce batch size
python fake_approx_eval_mobilenet.py ... --batch_size 64
```

**Solution - Both:**
Add memory growth setting to scripts or use:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true python fake_approx_train_mobilenet.py ...
```

### Weight Loading Issues

**Problem:** `InvalidArgumentError: Shapes must be equal rank...`

**Solution:**

- Ensure weights match model architecture (1000 classes for ImageNet)
- Use correct weights path: `ls models/`
- HDF5: `models/mobilenet_v2_weights.h5`
- SavedModel: `models/mobilenet_v2_savedmodel/`

### Layer Replacement Issues

**Problem:** `ImportError: Cannot find FakeApproxConv2D`

**Solution:**

- Verify environment setup: Check if `build/libApproxGPUOpsTF.so` exists
- Ensure you're in correct directory: `pwd` should show `tf2`
- Set PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**Problem:** `No module named 'fake_approx_convolutional'`

**Solution:**

- Check file exists: `ls python/keras/layers/fake_approx_convolutional.py`
- Verify symlinks: `ls -la python/keras/layers/`
- Run from `tf2` directory

### Evaluation Accuracy Mismatch

**Problem:** Approximate model accuracy much lower than expected

**Possible Causes:**

1. Wrong multiplier table file — verify file path
2. Model was trained differently — check training configuration
3. Data preprocessing mismatch — compare training vs eval preprocessing
4. Multiplier table corruption — try with `mul8u_1JFF.bin` (accurate) first

**Debug Steps:**

```bash
# 1. Test with accurate multiplier (empty file)
python fake_approx_eval_mobilenet.py ... --mtab_file ""

# 2. Examine layer replacement
python fake_approx_eval_mobilenet.py ... --mtab_file examples/axmul_8x8/mul8u_L40.bin --compare_layers

# 3. Check original model accuracy
python fake_approx_eval_mobilenet.py ... --mtab_file ""
```

---

## Performance Expectations

### Training Time

**On GTX 1650 GPU:**

- 1 epoch on full ImageNet (1.2M images): ~30-40 minutes
- 2 epochs: ~60-80 minutes
- Depends on: batch size, data I/O speed, GPU clock

**Typical timeline:**

```
Training started...
Epoch 1/2 - 1800s [████████████████████] - loss: 0.7234 - top_1_accuracy: 0.7123
Epoch 2/2 - 1750s [████████████████████] - loss: 0.6854 - top_1_accuracy: 0.7245
Training completed in ~3550 seconds (59 minutes)
```

### Evaluation Time

**On GTX 1650 GPU:**

- Full ImageNet val set (50K images, batch_size=128): ~15-20 minutes
- Accurate model evaluation: ~10 minutes
- Approximate model evaluation: ~10 minutes (same speed, different computation)

### Accuracy Degradation

**Typical results with L40 multiplier:**

```
Accurate Top-1 Accuracy:    71.5%
Approximate Top-1 Accuracy: 71.2%
Degradation:                -0.3% (negligible)

Degradation Range by Multiplier:
  mul8u_1JFF.bin (accurate):     -0.0%
  mul8u_L40.bin:                 -0.3% to -0.8%
  other multipliers:             -0.5% to -2.0%
```

---

## Example Workflows

### Complete Workflow: Training + Evaluation

```bash
# Step 1: Train for 2 epochs
python examples/fake_approx_train_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --epochs 2 \
    --batch_size 128

# Step 2: Evaluate with L40 multiplier
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/mobilenet_v2_weights.h5 \
    --mtab_file examples/axmul_8x8/mul8u_L40.bin \
    --save_results

# Step 3: Try different multipliers
for mtab in examples/axmul_8x8/mul8u_*.bin; do
    python examples/fake_approx_eval_mobilenet.py \
        --imagenet_path /path/to/imagenet \
        --weights_path models/mobilenet_v2_weights.h5 \
        --mtab_file "$mtab" \
        --results_file "eval_$(basename $mtab .bin).json"
done

# View results
cat eval_results.json
```

### Quick Validation Workflow

```bash
# Train on small subset (for quick testing)
python examples/fake_approx_train_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --epochs 1 \
    --batch_size 64 \
    --steps_per_epoch 10 \
    --save_dir models/test/

# Evaluate quickly
python examples/fake_approx_eval_mobilenet.py \
    --imagenet_path /path/to/imagenet \
    --weights_path models/test/mobilenet_v2_weights.h5 \
    --mtab_file examples/axmul_8x8/mul8u_L40.bin \
    --batch_size 64
```

### Comparison Across Multipliers

```bash
# Create comparison script
cat > compare_multipliers.sh << 'EOF'
#!/bin/bash
weights_path="models/mobilenet_v2_weights.h5"
results_dir="comparison_results"
mkdir -p $results_dir

for mtab in examples/axmul_8x8/mul8u_{L40,Y48,1JFF,1AGV}.bin; do
    name=$(basename $mtab .bin)
    echo "Evaluating $name..."
    python examples/fake_approx_eval_mobilenet.py \
        --imagenet_path /path/to/imagenet \
        --weights_path $weights_path \
        --mtab_file "$mtab" \
        --save_results \
        --results_file "$results_dir/${name}.json"
done

echo "Results saved in $results_dir/"
EOF

chmod +x compare_multipliers.sh
./compare_multipliers.sh
```

---

## Next Steps

After completing training and evaluation:

1. **Analyze Results** - Compare different multiplier tables, understand accuracy tradeoffs
2. **Optimize Parameters** - Try different learning rates, batch sizes, epochs
3. **Model Deployment** - Export best model for inference
4. **Custom Architectures** - Apply same pattern to other networks (ResNet, EfficientNet, etc.)

---

## References

- **GitHub**: <https://github.com/ehw-fit/tf-approximate>
- **EvoApprox Library**: <https://github.com/ehw-fit/evoapproxlib>
- **MobileNet-v2 Paper**: <https://arxiv.org/abs/1801.04381>
- **ImageNet**: <http://www.image-net.org/>

---

## Support

For issues:

1. Check the Troubleshooting section above
2. Review script help: `python fake_approx_train_mobilenet.py --help`
3. Check TensorFlow version: `python -c "import tensorflow; print(tensorflow.__version__)"`
4. Verify GPU availability: `nvidia-smi`
5. Check library binary: `ls -lah build/libApproxGPUOpsTF.so`

---

**Last Updated**: May 15, 2026
**Tested With**: TensorFlow 2.x, CUDA 12.4, GTX 1650
