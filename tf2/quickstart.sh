#!/bin/bash
# Quick start script to run tf-approximate examples
# This script sets up environment and runs the evaluation example

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========== TensorFlow Approximate Quick Start ==========${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TF2_DIR="${SCRIPT_DIR}"

if [ ! -d "$TF2_DIR" ]; then
    echo -e "${RED}Error: tf-approximate/tf2 directory not found!${NC}"
    exit 1
fi

echo -e "${YELLOW}Setting up environment...${NC}"
cd "$TF2_DIR"

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build directory not found. Please compile first:${NC}"
    echo "  cd $TF2_DIR/build"
    echo "  cmake .. -DTFAPPROX_CUDA_ARCHS=75 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4.1"
    echo "  make"
    exit 1
fi

# Check if libApproxGPUOpsTF.so exists
if [ ! -f "build/libApproxGPUOpsTF.so" ]; then
    echo -e "${RED}Error: libApproxGPUOpsTF.so not found in build directory!${NC}"
    exit 1
fi

# Setup environment variables
# Ensure the `python` subdirectory is on PYTHONPATH so packages like
# `keras.layers.fake_approx_convolutional` (located in python/keras/...) can be imported.
# Setup environment variables
# Do not prepend the local `python` directory to PYTHONPATH (it can conflict
# with the system `keras` package). Examples load the local FakeApproxConv2D
# by file path, so we don't need to modify PYTHONPATH here.
export PYTHONPATH="${PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TF2_DIR}/build"

echo -e "${GREEN}✓ Environment configured${NC}"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  Working directory: $TF2_DIR"

# Check if lenet5_weights exists
if [ ! -f "lenet5_weights" ]; then
    echo ""
    echo -e "${YELLOW}Training network (this will take ~5-10 minutes)...${NC}"
    python examples/fake_approx_train.py
    echo -e "${GREEN}✓ Training complete${NC}"
fi

echo ""
echo -e "${YELLOW}========== Test 1: Accurate Multiplication ==========${NC}"
python examples/fake_approx_eval.py

echo ""
echo -e "${YELLOW}========== Test 2: Approximate Multiplier (L40) ==========${NC}"
if [ -f "examples/axmul_8x8/mul8u_L40.bin" ]; then
    python examples/fake_approx_eval.py --mtab_file examples/axmul_8x8/mul8u_L40.bin
else
    echo -e "${RED}Warning: mul8u_L40.bin not found${NC}"
fi

echo ""
echo -e "${GREEN}========== All Tests Complete ==========${NC}"
echo ""
echo "For more information, see:"
echo "  - README_rpp.md in tf2/ directory"
echo "  - BUILD_FIXES_LOG.md in tf2/ directory"
