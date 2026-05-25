# SPRIM8 Bin Generation and Validation Plan

## Generation (`generate_sprim8_bins.py`)

1. **Discover models**: Scan `/home/rappy/workspace/Heidelberg_colab/Verilog/SPRIM8 & S_Exact8/Python` for all `SPRIM8_*.py` files.
2. **Dynamic import**: Load each module using `importlib.util` and locate the matching top-level function (e.g., `SPRIM8_41.py` → `SPRIM8_41()`).
3. **Compute table**: For all signed int8 input pairs `a, b ∈ [-128, 127]` (256 × 256 = 65,536 entries):
   - Call the multiplier function: `result = SPRIM8_XX(a, b)`
   - Normalize overflow to signed int16 range: `result = ((result + 2^15) % 2^16) - 2^15`
4. **Serialize**: Pack each 16-bit signed result as little-endian and write to `/mnt/new_ssd/workspace/tfapprox_rpp/tf-approximate/tf2/examples/axmul_8x8/fame/SPRIM8_XX.bin`.
5. **Output size**: Each `.bin` file is exactly 131,072 bytes (256 × 256 × 2 bytes per int16).

## Validation (`validate_sprim8_bins.py`)

The validation script independently verifies that each generated `.bin` table matches its behavioral model output:

1. **Load model & table**: For each multiplier, import the Python model and read the corresponding `.bin` file.
2. **Compare entries**: Iterate over all 65,536 signed input pairs in the same order as generation and compare:
   - Python output: `expected = SPRIM8_XX(a, b)` (normalized to int16 if needed)
   - Binary table value: `actual = table[index]` (unpacked as little-endian int16)
3. **Report mismatches**: If `expected ≠ actual`, log the mismatch with details (inputs and both values).
   - First 10 mismatches per model are printed; full count is always reported.
4. **Final summary**: Display pass/fail per model and an overall count (e.g., "35/35 models matched their .bin files").

### Usage

- Generate all multiplier tables: `python generate_sprim8_bins.py`
- Generate specific models: `python generate_sprim8_bins.py --model SPRIM8_41 --model SPRIM8_42`
- Validate all tables: `python validate_sprim8_bins.py`
- Validate specific models: `python validate_sprim8_bins.py --model SPRIM8_41`
