#!/usr/bin/env python3
"""Validate SPRIM8 lookup tables against their Python behavioral models.

This script compares each model's signed int8 output table against the
corresponding generated .bin file using the same signed input ordering as the
generator:
    a in [-128, 127]
    b in [-128, 127]

The .bin file is expected to store little-endian signed int16 values.
"""

from __future__ import annotations

import argparse
import importlib.util
import struct
from pathlib import Path
from types import ModuleType
from typing import Iterable, List, Sequence, Tuple


DEFAULT_SOURCE_DIR = Path(
    "/home/rappy/workspace/Heidelberg_colab/Verilog/SPRIM8 & S_Exact8/Python"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/new_ssd/workspace/tfapprox_rpp/tf-approximate/tf2/examples/axmul_8x8/fame"
)


def discover_model_files(source_dir: Path) -> List[Path]:
    return sorted(source_dir.glob("SPRIM8_*.py"))


def load_module(module_path: Path) -> ModuleType:
    module_name = f"validate_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_multiplier_function(module: ModuleType, function_name: str):
    try:
        function = getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module {module.__name__} does not define {function_name}"
        ) from exc

    if not callable(function):
        raise TypeError(
            f"{function_name} in {module.__name__} is not callable")

    return function


def read_bin_table(bin_path: Path) -> Sequence[int]:
    raw = bin_path.read_bytes()
    if len(raw) != 256 * 256 * 2:
        raise ValueError(
            f"{bin_path} has invalid size {len(raw)} bytes; expected 131072 bytes"
        )

    return struct.unpack("<65536h", raw)


def validate_single_model(module_path: Path, bin_dir: Path) -> Tuple[int, int, int]:
    module = load_module(module_path)
    multiplier_function = get_multiplier_function(module, module_path.stem)

    bin_path = bin_dir / f"{module_path.stem}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing bin file: {bin_path}")

    table_values = read_bin_table(bin_path)
    mismatches = 0
    checked = 0

    index = 0
    for a in range(-128, 128):
        for b in range(-128, 128):
            expected = int(multiplier_function(a, b))
            if expected < -32768 or expected > 32767:
                expected = ((expected + 2**15) % 2**16) - 2**15

            actual = table_values[index]
            if actual != expected:
                mismatches += 1
                if mismatches <= 10:
                    print(
                        f"  mismatch at a={a}, b={b}: py={expected}, bin={actual}"
                    )
            checked += 1
            index += 1

    return checked, mismatches, len(table_values)


def validate_models(source_dir: Path, bin_dir: Path, selected_models: Iterable[str]) -> int:
    selected = {name for name in selected_models} if selected_models else None
    model_files = discover_model_files(source_dir)

    if selected is not None:
        model_files = [path for path in model_files if path.stem in selected]

    if not model_files:
        raise FileNotFoundError(f"No SPRIM8 model files found in {source_dir}")

    total_models = 0
    failed_models = 0

    for module_path in model_files:
        total_models += 1
        print(f"Validating {module_path.stem} ...")
        checked, mismatches, table_len = validate_single_model(
            module_path, bin_dir)
        if mismatches:
            failed_models += 1
            print(
                f"  FAIL: checked {checked} entries, table_len={table_len}, mismatches={mismatches}"
            )
        else:
            print(f"  OK: checked {checked} entries, table_len={table_len}")

    print(
        f"Summary: {total_models - failed_models}/{total_models} models matched their .bin files"
    )
    return 1 if failed_models else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SPRIM8 .bin tables against Python behavioral models."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing SPRIM8_*.py behavioral model files.",
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing generated SPRIM8 .bin files.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
        help="Optional SPRIM8 model stem to validate. Repeat to select multiple models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return validate_models(args.source_dir, args.bin_dir, args.models)


if __name__ == "__main__":
    raise SystemExit(main())
