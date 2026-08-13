"""Kernel-loop targets for NCU profiling of the Step-4 B anchors.

Per the §14 Step-4 charter: profile the LARGE-RANK anchor (rank 128
prefill, both families) and the DEVICE-REVERSAL anchor (down/r128/decode
— the one cell where the tuned one-launch kernel (then named fused_flat)
only ties tuned stock on H200 while winning 1.18x on GB300). Prints
revision + content digest so the log is self-describing.

Usage (under ncu)::

    ncu --set basic --section WarpStateStats --section LaunchStats \
        --section MemoryWorkloadAnalysis \
        -k "regex:_one_launch_sliced_lora_b_kernel|fused_moe_kernel" \
        --launch-skip 8 --launch-count 4 -o report -f \
        python3 -m benchmark.kernels.lora_moe.ncu_targets_b \
        --family one_launch --site gate_up --cell prefill \
        --config bn128-bk16-w4-s3
"""

from __future__ import annotations

import argparse

import torch

from benchmark.kernels.lora_moe.bench_common import parse_table_config
from benchmark.kernels.lora_moe.bench_lora_b import _BFixture
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.timing import (
    content_fingerprint,
    resolve_source_revision,
)

CELLS = {"decode": 64, "prefill": 2048}
ITERATIONS = 16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        required=True,
        choices=("stock", "lean_two_launch", "one_launch"),
    )
    parser.add_argument("--site", required=True, choices=("gate_up", "down"))
    parser.add_argument("--cell", required=True, choices=tuple(CELLS))
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument(
        "--config",
        required=True,
        help="the PROMOTED config for this device/site/regime/family, "
        "mined from that device's own B table",
    )
    arguments = parser.parse_args()
    device = torch.device("cuda")
    print(f"revision: {resolve_source_revision()}")
    print(f"source_digest: {content_fingerprint()}")
    config = parse_table_config(arguments.config)
    print(f"resolved config: {config}")

    case = build_case(
        device="cuda",
        model_preset="qwen35_35b",
        topology=Topology(tp_size=8, ep_size=8),
        adapter_cell=AdapterCell(
            active_adapters=4, include_base_rows=True, slot_capacity=8
        ),
        route_generator="iid",
        num_tokens=CELLS[arguments.cell],
        active_rank=arguments.rank,
        seed=11,
        source_revision="ncu-target-b",
    )
    fixture = _BFixture(case, device)
    torch.cuda.synchronize()
    for _ in range(ITERATIONS):
        fixture.run_family(arguments.site, arguments.family, config)
    torch.cuda.synchronize()
    print(f"{arguments.family}/{arguments.site}/{arguments.cell}: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
