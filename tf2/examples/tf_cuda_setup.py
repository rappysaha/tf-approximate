"""Helpers for configuring TensorFlow's CUDA runtime search path."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


def _existing_paths(paths):
    return [path for path in paths if path and os.path.isdir(path)]


def configure_cuda_runtime():
    """Prepend local CUDA and NVIDIA wheel library directories to LD_LIBRARY_PATH."""
    tf2_root = Path(__file__).resolve().parents[1]
    cuda_roots = [
        os.environ.get('CUDA_HOME'),
        '/usr/local/cuda-12.4.1',
        '/usr/local/cuda',
    ]

    lib_dirs = [str(tf2_root / 'build')]
    for cuda_root in _existing_paths(cuda_roots):
        lib_dirs.extend([
            os.path.join(cuda_root, 'targets', 'x86_64-linux', 'lib'),
            os.path.join(cuda_root, 'targets', 'x86_64-linux', 'lib', 'stubs'),
        ])

    site_packages = Path(sys.prefix) / 'lib'
    if site_packages.exists():
        for wheel_lib_dir in glob.glob(str(site_packages / 'python*' / 'site-packages' / 'nvidia' / '*' / 'lib')):
            lib_dirs.append(wheel_lib_dir)

    lib_dirs = _existing_paths(lib_dirs)

    current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    merged_paths = []

    for path in lib_dirs:
        if path not in merged_paths:
            merged_paths.append(path)

    if current_ld_library_path:
        for path in current_ld_library_path.split(':'):
            if path and path not in merged_paths:
                merged_paths.append(path)

    if merged_paths:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(merged_paths)
