#!/usr/bin/env python3
"""Generate 8x8 approximate-multiplier lookup tables from SPRIM8 models.

Each behavioral model in the source directory is expected to expose a top-level
function with the same name as the file stem, for example `SPRIM8_41.py`
defines `SPRIM8_41(a, b)`.
"""

from __future__ import annotations

import argparse
import importlib.util
import struct
from pathlib import Path
from types import ModuleType
from typing import Iterable, List


DEFAULT_SOURCE_DIR = Path(
    "/home/rappy/workspace/Heidelberg_colab/Verilog/SPRIM8 & S_Exact8/Python"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/new_ssd/workspace/tfapprox_rpp/tf-approximate/tf2/examples/axmul_8x8/fame"
)


def discover_model_files(source_dir: Path) -> List[Path]:
    return sorted(source_dir.glob("SPRIM8_*.py"))


def load_module(module_path: Path) -> ModuleType:
    module_name = f"sprim8_{module_path.stem}"
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


def generate_table(multiplier_function) -> bytes:
    table = bytearray()
    pack_int16 = struct.Struct("<h").pack

    for a in range(-128, 128):
        for b in range(-128, 128):
            value = int(multiplier_function(a, b))
            if value < -32768 or value > 32767:
                value = ((value + 2**15) % 2**16) - 2**15
            table.extend(pack_int16(value))

    return bytes(table)


def write_table(output_path: Path, table_bytes: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(table_bytes)


def generate_tables(source_dir: Path, output_dir: Path, selected_models: Iterable[str]) -> None:
    selected = {name for name in selected_models} if selected_models else None
    model_files = discover_model_files(source_dir)

    if selected is not None:
        model_files = [path for path in model_files if path.stem in selected]

    if not model_files:
        raise FileNotFoundError(
            f"No SPRIM8 model files found in {source_dir}"
        )

    for module_path in model_files:
        module = load_module(module_path)
        multiplier_function = get_multiplier_function(module, module_path.stem)
        table_bytes = generate_table(multiplier_function)
        output_path = output_dir / f"{module_path.stem}.bin"
        write_table(output_path, table_bytes)
        print(f"Wrote {output_path} ({len(table_bytes)} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate binary lookup tables from SPRIM8 behavioral models."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing SPRIM8_*.py behavioral model files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated .bin files will be written.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
        help="Optional SPRIM8 model stem to generate. Repeat to select multiple models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_tables(args.source_dir, args.output_dir, args.models)


if __name__ == "__main__":
    main()
