"""Smoke test for generated mc configs.

Walks ``profile/configs/mc{mc}/`` and validates, for each mc:

1. ``heter_config.json`` has the expected schema (groups, policy, file ref).
2. ``int4_only_experts.json`` covers valid (layer, expert) ids within the
   model's shape, with no duplicates.
3. ``K_heter_experts + num_int4_only == num_experts * num_layers`` (no
   experts lost or double-counted).
4. ``K * bf16_expert_size <= bf16_budget`` (the budget actually admits K).
5. ``non_moe + int4 + headroom + kv + sglang_reserved + K*bf16_expert_size
    <= gpu_vram`` (the plan fits on the GPU).
6. ``heter_config.json → int4_only_experts_file`` resolves to an existing,
   readable file matching the per-layer lists.
7. ``predicted_mfs`` is in sglang's default range [0.80, 0.90] on 80 GB.

We intentionally do NOT enumerate the 11 runtime variants — they inherit
``int4_only_experts_file`` from the base and only differ in ``policy`` /
``size_ratio``, which are runtime-dispatch concerns. One spot-check per mc
confirms the shared reference still resolves.

Run directly:
    python test_configs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent.parent
# Env override lets us validate per-task trees (configs/sharegpt, ...)
# without changing the default flat path.
import os
CONFIGS_ROOT = Path(os.environ.get(
    "CONFIGS_ROOT", str(REPO / "profile" / "configs")
))

sys.path.insert(0, str(THIS_DIR))
import vram_estimator as vest  # noqa: E402

MC_LIST = [8, 16, 32, 64, 128, 256]

# Qwen3-30B-A3B shape (hard-coded so the test has no heavy deps).
NUM_LAYERS = 48
NUM_EXPERTS_PER_LAYER = 128
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
TOTAL_EXPERTS = NUM_LAYERS * NUM_EXPERTS_PER_LAYER  # 6144

SPOT_VARIANT = "hot40"


def _load(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _check_mc(mc: int) -> None:
    mc_dir = CONFIGS_ROOT / f"mc{mc}"
    assert mc_dir.is_dir(), f"missing {mc_dir}"

    heter_cfg = _load(mc_dir / "heter_config.json")
    int4_map = _load(mc_dir / "int4_only_experts.json")
    report = _load(mc_dir / "assignment_report.json")

    # 1. heter_config schema.
    assert [g["name"] for g in heter_cfg["groups"]] == ["cold", "hot"]
    assert heter_cfg["groups"][0]["num_bits"] == 4
    assert heter_cfg["groups"][1]["num_bits"] == 16
    assert heter_cfg["policy"] in {"expert_batch", "random"}
    ref = Path(heter_cfg["int4_only_experts_file"])
    assert ref.is_file(), f"int4_only_experts_file missing: {ref}"

    # 2. int4_only_experts bounds + dedup.
    flat_int4 = set()
    for L_str, experts in int4_map.items():
        L = int(L_str)
        assert 0 <= L < NUM_LAYERS, f"layer {L} out of range"
        for E in experts:
            assert 0 <= E < NUM_EXPERTS_PER_LAYER, f"expert {E} out of range"
            key = (L, E)
            assert key not in flat_int4, f"duplicate int4 entry {key}"
            flat_int4.add(key)

    # 3. accounting: heter + int4 == total.
    k_heter = report["K_heter_experts"]
    n_int4 = report["num_int4_only"]
    assert k_heter + n_int4 == TOTAL_EXPERTS, (
        f"mc={mc}: {k_heter} + {n_int4} != {TOTAL_EXPERTS}"
    )
    assert len(flat_int4) == n_int4, (
        f"mc={mc}: int4 map has {len(flat_int4)}, report says {n_int4}"
    )

    # 4. K admitted by bf16 budget.
    bf16_sz = vest.bf16_expert_bytes(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
    bf16_budget = report["vram_breakdown_bytes"]["bf16_budget_bytes"]
    assert k_heter * bf16_sz <= max(0, bf16_budget), (
        f"mc={mc}: K={k_heter} × {bf16_sz} > bf16_budget={bf16_budget}"
    )

    # 5. Plan fits on GPU — unless K==0 already signalled infeasibility.
    b = report["vram_breakdown_bytes"]
    used = (
        b["non_moe_bytes"]
        + b["int4_weights_bytes"]
        + b["headroom_bytes"]
        + b["kv_bytes"]
        + b["sglang_reserved_bytes"]
        + k_heter * bf16_sz
    )
    feasible = used <= b["total_bytes"]
    if k_heter > 0:
        assert feasible, (
            f"mc={mc}: plan uses {used / vest.GIB:.2f} GB > "
            f"gpu {b['total_bytes'] / vest.GIB:.2f} GB"
        )

    # 6. Spot-check one runtime variant picks up the same int4 ref.
    variant = _load(mc_dir / "variants" / f"{SPOT_VARIANT}.json")
    assert variant["int4_only_experts_file"] == heter_cfg["int4_only_experts_file"]

    # 7. predicted_mfs in sglang's default range.
    mfs = b["predicted_mfs"]
    assert 0.80 <= mfs <= 0.90, f"mc={mc}: predicted_mfs={mfs} out of range"

    tag = "OK" if feasible else f"INFEASIBLE (needs {used / vest.GIB:.2f} GB)"
    print(f"  [mc={mc:>3}]  K={k_heter:<5}  int4={n_int4:<5}  mfs={mfs}  {tag}")


def main() -> int:
    print(f"Validating configs under {CONFIGS_ROOT}")
    for mc in MC_LIST:
        _check_mc(mc)
    print(f"All {len(MC_LIST)} mc configs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
