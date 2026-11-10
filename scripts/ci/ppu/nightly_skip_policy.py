#!/usr/bin/env python3
"""Exact, documented skip policy for the PPU nightly strict runner."""

import argparse
import json


def allowed_skips():
    batch_sizes = [2**x for x in range(12)]
    batch_sizes += [x + 1 for x in batch_sizes]
    batch_sizes[1] = "2_0"
    batch_sizes[12] = "2_1"
    skips = [
        f"registered.kernels.ops.attention.test_rope.test_partial_rope[64-{rope_dim}-{is_neox}-{batch_size}]"
        for rope_dim in (80, 96, 128)
        for is_neox in (False, True)
        for batch_size in batch_sizes
    ]
    skips.append(
        "registered.kernels.ops.diffusion.test_timestep_embedding.test_timestep_embedding_perf"
    )
    return skips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    skips = allowed_skips()
    if args.json:
        print(json.dumps(skips, indent=2))
    else:
        for node in skips:
            print(f"--allow-skip\n{node}")


if __name__ == "__main__":
    main()
