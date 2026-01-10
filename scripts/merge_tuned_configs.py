#!/usr/bin/env python3
"""Merge individual batch-size tuning results into combined config files.

This script is used by the auto-tune workflow to merge configs from multiple
batch size groups into a single config file per model/runner combination.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def merge_configs(input_dir: str, output_dir: str) -> None:
    """Merge individual batch size configs into combined files.

    Args:
        input_dir: Directory containing downloaded artifacts with per-group configs.
        output_dir: Directory to write merged config files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Group configs by (triton_version_dir, filename)
    # Structure: {rel_path: {batch_size: config}}
    configs_by_file: dict[str, dict[int, dict]] = defaultdict(dict)

    # Find all JSON config files
    json_files = list(input_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    for json_file in json_files:
        # Skip checkpoint files
        if ".checkpoint" in json_file.name:
            print(f"  Skipping checkpoint file: {json_file.name}")
            continue

        # Find triton_* directory in path to get relative path
        # e.g., triton_3_0_0/E=8,N=4096,dtype=fp16.json
        parts = json_file.parts
        try:
            triton_idx = next(
                i for i, p in enumerate(parts) if p.startswith("triton_")
            )
            rel_path = "/".join(parts[triton_idx:])
        except StopIteration:
            # No triton dir found, use filename directly
            rel_path = json_file.name

        # Load config file
        try:
            with open(json_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Warning: Failed to parse {json_file}: {e}")
            continue

        # Merge batch sizes from this file
        # Data format: {batch_size_str: config_dict}
        for batch_size_str, config in data.items():
            try:
                batch_size = int(batch_size_str)
                configs_by_file[rel_path][batch_size] = config
            except ValueError:
                print(f"  Warning: Invalid batch size '{batch_size_str}' in {json_file}")
                continue

    if not configs_by_file:
        print("No config files found to merge")
        return

    # Write merged configs
    for rel_path, batch_configs in configs_by_file.items():
        # Sort by batch size for consistent output
        sorted_configs = {
            str(bs): config for bs, config in sorted(batch_configs.items())
        }

        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w") as f:
            json.dump(sorted_configs, f, indent=4)
            f.write("\n")

        print(f"Merged {len(sorted_configs)} batch sizes into {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge individual batch-size tuning results into combined config files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing downloaded artifacts with per-group configs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write merged config files",
    )
    args = parser.parse_args()

    merge_configs(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
