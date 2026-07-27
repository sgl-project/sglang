"""Kernel-loop targets for NCU profiling of the Step-3 promoted winners.

Runs ONE promoted kernel in a short eager loop so ``ncu`` can attach and
capture it (section 11.2 obligation). No timing here — NCU is the
measurement.

Ninth S3 review: the first version hardcoded one grouped config for every
device/site/regime (wrong for five of eight targets against the archived
sweeps) and only profiled the serving16 SGMV fallback. Now:

* grouped targets take ``--grouped-config bnX-bkY-wW-sS`` — the caller
  passes the PROMOTED config mined from that device's own sweep archive;
* SGMV targets are the per-regime PROPOSAL plus the fallback:
  ``sgmv_unchunked_decode`` (proposed decode variant),
  ``sgmv_chunked128_prefill`` (proposed prefill variant),
  ``sgmv16_decode`` / ``sgmv16_prefill`` (single-variant fallback);
* the resolved config and revision are printed so the profiling log is
  self-describing.

Usage (under ncu; 4 captured launches after 8 warmups)::

    ncu --set basic --section WarpStateStats --section LaunchStats \
        -k "regex:_grouped_lora_a_kernel|_chunked_lora_shrink|_sgemm_lora_a" \
        --launch-skip 8 --launch-count 4 -o report -f \
        python3 -m benchmark.kernels.lora_moe.ncu_targets \
        --target grouped_gate_decode --grouped-config bn32-bk64-w4-s4
"""

from __future__ import annotations

import argparse

import torch

from benchmark.kernels.lora_moe.bench_lora_a import _LegFixture
from benchmark.kernels.lora_moe.bench_sgmv_real import (
    SERVING_CHUNK,
    synthesize_chunked_batch_info,
    synthesize_unchunked_batch_info,
)
from benchmark.kernels.lora_moe.bench_shared_dedup import _SharedFixture
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.timing import (
    content_fingerprint,
    resolve_source_revision,
)
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import (
    chunked_sgmv_lora_shrink_forward,
)
from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a
from sglang.srt.lora.sgl_lora.routing import ROUTE_ALIGNED

CELLS = {"decode": (64, 16), "prefill": (2048, 64)}
ITERATIONS = 16
SGMV_TARGETS = (
    "sgmv16_decode",
    "sgmv16_prefill",
    "sgmv_unchunked_decode",
    "sgmv_chunked128_prefill",
    # Tenth S3 review: decode qualification must include the GRAPH-STABLE
    # serving geometries (capacity-padded grids), not only the compact lab
    # forms — a captured decode graph launches the padded grid.
    "sgmv16_graph_decode",
    "sgmv_unchunked_stable_decode",
)


def _parse_grouped_config(text: str) -> dict:
    fields = {"GROUP_SIZE_M": 8}
    for piece in text.split("-"):
        if piece.startswith("bn"):
            fields["BLOCK_SIZE_N"] = int(piece[2:])
        elif piece.startswith("bk"):
            fields["BLOCK_SIZE_K"] = int(piece[2:])
        elif piece.startswith("w"):
            fields["num_warps"] = int(piece[1:])
        elif piece.startswith("s"):
            fields["num_stages"] = int(piece[1:])
    missing = {"BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"} - set(fields)
    if missing:
        raise SystemExit(f"cannot parse grouped config {text!r}: missing {missing}")
    return fields


def _case(num_tokens, rank, shared):
    return build_case(
        device="cuda",
        model_preset="qwen35_35b",
        topology=Topology(tp_size=8, ep_size=8),
        adapter_cell=AdapterCell(
            active_adapters=4, include_base_rows=True, slot_capacity=8
        ),
        route_generator="iid",
        num_tokens=num_tokens,
        active_rank=rank,
        shared_factor_signature="shared_gate_up_a" if shared else "per_expert",
        seed=11,
        source_revision="ncu-target",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        required=True,
        choices=[
            f"grouped_{site}_{cell}" for site in ("gate", "down") for cell in CELLS
        ]
        + list(SGMV_TARGETS),
    )
    parser.add_argument(
        "--grouped-config",
        default=None,
        help="bnX-bkY-wW-sS — REQUIRED for grouped targets; pass the "
        "promoted config mined from THIS device's sweep archive",
    )
    arguments = parser.parse_args()
    device = torch.device("cuda")
    print(f"revision: {resolve_source_revision()}")
    print(f"source_digest: {content_fingerprint()}")

    if arguments.target.startswith("grouped"):
        if arguments.grouped_config is None:
            raise SystemExit(
                "grouped targets need --grouped-config (the promoted config "
                "for this device/site/regime; ninth S3 review)"
            )
        config = _parse_grouped_config(arguments.grouped_config)
        print(f"resolved grouped config: {config}")
        _, site_name, cell = arguments.target.split("_")
        num_tokens, rank = CELLS[cell]
        fixture = _LegFixture(_case(num_tokens, rank, shared=False), device)
        aligned = fixture.route(ROUTE_ALIGNED)
        site = "gate_up" if site_name == "gate" else "down"
        inp, weight, out = fixture.site_buffers(site)
        torch.cuda.synchronize()
        for _ in range(ITERATIONS):
            grouped_lora_a(
                inp, weight, out, aligned, config=config, pair_input=site == "down"
            )
        torch.cuda.synchronize()
    else:
        cell = arguments.target.rsplit("_", 1)[1]
        num_tokens, rank = CELLS[cell]
        case = _case(num_tokens, rank, shared=True)
        fixture = _SharedFixture(case, device)
        weights = fixture.base.a_gate_up.view(
            case.slot_capacity, -1, case.moe_hidden_size
        )
        common = dict(
            max_loras=case.slot_capacity,
            physical_rank=case.physical_rank,
            device=device,
        )
        if arguments.target.startswith("sgmv_unchunked"):
            stable = "stable" in arguments.target
            info = synthesize_unchunked_batch_info(
                fixture.base.token_slots,
                capacity_segments=case.slot_capacity if stable else None,
                max_len_ceiling=case.num_tokens if stable else None,
                **common,
            )
            runner = lambda: sgemm_lora_a_fwd(  # noqa: E731
                fixture.base.hidden_states, weights, info, 2
            )
            print(
                "resolved sgmv variant: unchunked"
                + (" capacity-stable" if stable else "")
                + " (BLOCK_S16/BLOCK_K256/BLOCK_R16)"
            )
        else:
            chunk = 128 if "chunked128" in arguments.target else SERVING_CHUNK
            graph = "graph" in arguments.target
            info = synthesize_chunked_batch_info(
                fixture.base.token_slots,
                chunk=chunk,
                graph_capacity=num_tokens if graph else None,
                **common,
            )
            runner = lambda: chunked_sgmv_lora_shrink_forward(  # noqa: E731
                fixture.base.hidden_states, weights, info, 2
            )
            print(
                f"resolved sgmv variant: chunked, chunk={chunk}"
                + (f", graph_capacity={num_tokens}" if graph else "")
            )
        torch.cuda.synchronize()
        for _ in range(ITERATIONS):
            runner()
        torch.cuda.synchronize()
    print(f"{arguments.target}: {ITERATIONS} launches done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
