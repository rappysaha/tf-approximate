Plan: Build `libApproxGPUOpsTF.so` on Jetson Orin Nano

Native build on Jetson Orin Nano is the recommended path. It minimizes cross-architecture issues, lets CMake discover local TensorFlow and CUDA correctly, and is easier to validate end-to-end on the target device.

Start order

Yes: first transport the code to `gicon1`, then build there. Do not try to build on the local workstation and copy only the binary unless you are intentionally doing a cross-compile workflow. This repo’s build expects the target machine to discover its own CUDA and TensorFlow environment.

Recommended steps

1. Prepare the Orin Nano environment on `gicon1`.

- Confirm the JetPack/CUDA version, Python version, and available memory/swap.
- Confirm TensorFlow is installed in the Python environment that will be used for the build.
- Confirm `nvcc` resolves from the CUDA toolkit on the server.
- Confirm the GPU architecture target. For Orin Nano, use SM 87 unless the server reports a different compute capability.

1. Copy or sync the repository to `gicon1`.

- Transfer the current `tf-approximate` source tree to the Jetson.
- Keep the source tree on the Jetson itself so CMake and TensorFlow discovery happen against the native environment.
- If you already use git on the server, `git pull` there is fine; otherwise use `rsync` or `scp`.

1. Create a clean build directory on `gicon1`.

- Work from `tf2/build`.
- Remove or isolate any stale build artifacts before reconfiguring.
- Make sure the build uses the server’s Python, CUDA, and TensorFlow environment.

1. Configure CMake for the Orin Nano build.

- Use the existing CMake flow in `tf2`.
- Set `TFAPPROX_CUDA_ARCHS=87` for Orin Nano.
- Point CMake at the server CUDA root if auto-discovery does not pick it up.
- Keep CPU fallback enabled unless you explicitly want a GPU-only build.

1. Compile the shared library.

- Run `cmake ..` and then `make` in the build directory.
- Capture any missing include/library issues in the log.
- Expect the main artifact to be `build/libApproxGPUOpsTF.so`.

1. Patch Jetson-specific runtime path assumptions if needed.

- Update `tf_cuda_setup.py` if it assumes x86_64 CUDA paths.
- Remove hardcoded desktop CUDA paths from `quickstart.sh` if they do not match the Jetson layout.
- Keep these changes minimal and only apply them if the native build or runtime check fails because of path assumptions.

1. Verify the built artifact on `gicon1`.

- Confirm the shared object is ARM64.
- Confirm it links against the Jetson CUDA/TensorFlow libraries.
- Load it with `tf.load_op_library`.

1. Run smoke tests.

- Execute `fake_approx_eval.py` in accurate mode first.
- Run it again with one multiplier table to confirm approximate mode still works.
- Stop after basic loader and execution validation; do not start tuning performance yet.

1. Document the final Jetson workflow.

- Add a Jetson section to `README_rpp.md`.
- Record the exact build command, CUDA root, TensorFlow version, and any Jetson-only fixes.

Relevant files

- `CMakeLists.txt`
- `FindTensorflow.cmake`
- `tf_cuda_setup.py`
- `quickstart.sh`
- `README_rpp.md`
- `fake_approx_eval.py`

Verification

- Shared object exists and reports ARM64 ELF.
- Dynamic dependencies resolve to aarch64 CUDA/TensorFlow libs.
- TensorFlow loads the plugin successfully.
- GPU is visible to TensorFlow.
- Evaluation script runs in accurate mode and approximate mode without loader or link failures.

Scope boundaries

- Included: native Jetson build plan, runtime compatibility fixes, validation workflow.
- Excluded: cross-compilation toolchains, containerization, and model performance tuning.
