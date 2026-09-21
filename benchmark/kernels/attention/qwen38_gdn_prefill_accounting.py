"""Static semantic-traffic accounting for Qwen3.8 TP1 GDN prefill."""

import argparse
import json
import math

BF16_BYTES = 2
FP32_BYTES = 4
Q_WIDTH = 2048
K_WIDTH = 2048
V_WIDTH = 6144
Z_WIDTH = 6144
GATE_WIDTH = 48
QKV_WIDTH = Q_WIDTH + K_WIDTH + V_WIDTH
LAYERS = 48
SCHEDULER_CHUNK = 16384


def _per_token() -> dict[str, dict[str, int]]:
    # Exact pinned SGLANG_USE_AITER ratio-3 path:
    #   cat Q/K/V + contiguous B/A, Conv1D output, contiguous Q/K/V,
    #   and FP32 g/beta. Q/K L2Norm stays inside the scan in both paths.
    pinned_allocated = (
        QKV_WIDTH * BF16_BYTES
        + 2 * GATE_WIDTH * BF16_BYTES
        + QKV_WIDTH * BF16_BYTES
        + QKV_WIDTH * BF16_BYTES
        + 2 * GATE_WIDTH * FP32_BYTES
    )
    fused_allocated = QKV_WIDTH * BF16_BYTES + 2 * GATE_WIDTH * FP32_BYTES

    # Reads + writes at semantic tensor widths; parameter-vector reads in the
    # gating calculation are counted per token, matching the kernel contract.
    gating_traffic = (
        2 * GATE_WIDTH * BF16_BYTES
        + 2 * GATE_WIDTH * FP32_BYTES
        + 2 * GATE_WIDTH * FP32_BYTES
    )
    pinned_traffic = (
        2 * QKV_WIDTH * BF16_BYTES
        + 4 * GATE_WIDTH * BF16_BYTES
        + 2 * QKV_WIDTH * BF16_BYTES
        + 2 * QKV_WIDTH * BF16_BYTES
        + gating_traffic
    )
    fused_traffic = 2 * QKV_WIDTH * BF16_BYTES + gating_traffic
    views_traffic = (
        2 * QKV_WIDTH * BF16_BYTES + 2 * QKV_WIDTH * BF16_BYTES + gating_traffic
    )
    views_allocated = (
        QKV_WIDTH * BF16_BYTES + QKV_WIDTH * BF16_BYTES + 2 * GATE_WIDTH * FP32_BYTES
    )
    fused_unpack_traffic = (
        2 * (QKV_WIDTH + Z_WIDTH + 2 * GATE_WIDTH) * BF16_BYTES
        + 2 * QKV_WIDTH * BF16_BYTES
        + 2 * QKV_WIDTH * BF16_BYTES
        + gating_traffic
    )
    fused_unpack_allocated = (
        (QKV_WIDTH + Z_WIDTH + 2 * GATE_WIDTH) * BF16_BYTES
        + QKV_WIDTH * BF16_BYTES
        + QKV_WIDTH * BF16_BYTES
        + 2 * GATE_WIDTH * FP32_BYTES
    )

    return {
        "pinned": {
            "semantic_bytes": pinned_traffic,
            "allocated_bytes": pinned_allocated,
            "launches": 8,
            "allocations": 9,
        },
        "fused": {
            "semantic_bytes": fused_traffic,
            "allocated_bytes": fused_allocated,
            "launches": 1,
            "allocations": 5,
        },
        "removed": {
            "semantic_bytes": pinned_traffic - fused_traffic,
            "allocated_bytes": pinned_allocated - fused_allocated,
            "launches": 7,
            "allocations": 4,
        },
        "alternate_baselines": {
            "projection_views": {
                "semantic_bytes": views_traffic,
                "allocated_bytes": views_allocated,
                "launches": 5,
                "allocations": 6,
            },
            "fused_unpack": {
                "semantic_bytes": fused_unpack_traffic,
                "allocated_bytes": fused_unpack_allocated,
                "launches": 6,
                "allocations": 10,
            },
        },
    }


def _shape(tokens: int, layers: int, scheduler_chunk: int) -> dict[str, object]:
    per_token = _per_token()
    scheduler_passes = math.ceil(tokens / scheduler_chunk)
    removed = per_token["removed"]
    return {
        "tokens": tokens,
        "layers": layers,
        "scheduler_chunk": scheduler_chunk,
        "scheduler_passes": scheduler_passes,
        "removed": {
            "semantic_bytes": removed["semantic_bytes"] * tokens * layers,
            "semantic_gib": (removed["semantic_bytes"] * tokens * layers / (1024**3)),
            "allocated_bytes": removed["allocated_bytes"] * tokens * layers,
            "allocated_gib": (removed["allocated_bytes"] * tokens * layers / (1024**3)),
            "launches": removed["launches"] * layers * scheduler_passes,
            "allocation_objects": (removed["allocations"] * layers * scheduler_passes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 60000, 131072])
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--scheduler-chunk", type=int, default=SCHEDULER_CHUNK)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "per_token": _per_token(),
                "shapes": [
                    _shape(tokens, args.layers, args.scheduler_chunk)
                    for tokens in args.tokens
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
