# SPDX-License-Identifier: Apache-2.0
"""Complete-front graph-replay benchmark, not an end-to-end gain estimate."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import triton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    test = (
        Path(__file__).resolve().parents[2]
        / "test/registered/amd/test_shared_router_gfx950.py"
    )
    spec = importlib.util.spec_from_file_location("reference", test)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rows = []
    for m in (6, 12):
        inputs = ref.operands(m)
        packed_down = ref.pack_reference_down(inputs[4])
        ref.compare(ref.shared_router(*inputs), ref.native(inputs, packed_down))
        native_us, fused_us = [], []
        for repeat in range(5):
            calls = [
                (native_us, lambda: ref.native(inputs, packed_down)),
                (fused_us, lambda: ref.shared_router(*inputs)),
            ]
            if repeat % 2:
                calls.reverse()
            for samples, call in calls:
                samples.append(1000 * triton.testing.do_bench_cudagraph(call, rep=200))
        rows.append(dict(m=m, native_us=native_us, fused_us=fused_us))
    args.output.write_text(
        json.dumps(dict(device=torch.cuda.get_device_name(), results=rows), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
