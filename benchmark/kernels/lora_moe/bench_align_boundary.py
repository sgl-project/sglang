"""Site the aligned-view kernel boundary: JIT CUDA vs fused Triton.

Redo of the distrusted single-seed crossover (execution plan section 39's
24576). Every defect the methodology audit identified is addressed:

* **Rung-edge-aware V grid.** The JIT path has three config thresholds, each
  flanked so a step is attributed to its cause: EPT 8->16 at V=8191|8192, the
  48 KiB dynamic-smem opt-in at V=12255|12256 (host-side, per-launch
  `cudaFuncSetAttribute`), and EPT 16->32 at V=16383|16384 — where 16352 and
  16384 share smem size but differ in EPT, isolating the rung effect.
* **Seeds, interleaved repeats, and a winner gate.** Three route seeds x two
  interleaved repeats per arm (arm order alternates within a cell, so clock
  drift cannot systematically favor one arm). An arm WINS a cell only if it is
  faster in every (seed, repeat) sample and the geometric-mean margin is at
  least MIN_MARGIN. Anything else is TIED, and a boundary may only be sited
  between decided cells.
* **Both timing modes.** Graph mode (hot L2, CPU cost excluded) models decode;
  eager mode (flushed L2, CPU launch cost INCLUDED) models prefill — and is
  where the fused arm's three Triton launches pay real Python cost against the
  JIT arm's single FFI call. The policy constant governs both, so both are
  measured and reported per cell.
* **Two route families.** iid everywhere; hotset (80% of pairs on 8 experts)
  across the contested band, since atomic contention depends on bucket skew.
* **Direct-call arms.** The literal branch bodies of
  `build_virtual_expert_routing`: jit = `_build_virtual_topk_ids` +
  `_align_block_size_jit` (charged its ID pass), fused = the kernel that
  derives the key inline. No monkey-patching.

Usage:
    python3 -m benchmark.kernels.lora_moe.bench_align_boundary \
        --output align_boundary_v1.json --source-revision <sha>
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ROUTE_INCLUSIVE,
    measure,
    new_suite,
    write_suite,
)
from sglang.kernels.ops.moe.virtual_experts import _align_block_size_jit
from sglang.srt.lora.sgl_lora.fused_align import fused_align_block_size
from sglang.srt.lora.sgl_lora.routing import (
    _build_virtual_topk_ids,
    _routing_capacity,
)

BLOCK_SIZE_M = 16
TOP_K = 8
SLOT_CAPACITY = 32

# All multiples of 32 (E = V / 32 with L_cap = 32, so bucket occupancy is
# comparable across cells). Flanking pairs per threshold — see module docstring.
V_GRID = (
    1024,
    4096,
    8160,
    8192,
    10240,
    12224,
    12288,
    14336,
    16352,
    16384,
    18432,
    20480,
    24576,
    28672,
    32736,
)
HOTSET_V = (8192, 12288, 14336, 16384, 20480, 24576)
P_GRID = (8, 2048, 16384)  # T = P / TOP_K
SEEDS = (11, 137, 997)
REPEATS = 2
HOTSET_EXPERTS = 8
HOTSET_FRACTION = 0.8


def _route(
    family: str, lora_experts_per_adapter: int, num_tokens: int, seed: int, device
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if family == "iid":
        topk_ids = torch.randint(
            0,
            lora_experts_per_adapter,
            (num_tokens, TOP_K),
            generator=generator,
            dtype=torch.int32,
        )
    elif family == "hotset":
        hot = torch.randint(
            0,
            min(HOTSET_EXPERTS, lora_experts_per_adapter),
            (num_tokens, TOP_K),
            generator=generator,
            dtype=torch.int32,
        )
        cold = torch.randint(
            0,
            lora_experts_per_adapter,
            (num_tokens, TOP_K),
            generator=generator,
            dtype=torch.int32,
        )
        take_hot = (
            torch.rand((num_tokens, TOP_K), generator=generator) < HOTSET_FRACTION
        )
        topk_ids = torch.where(take_hot, hot, cold)
    else:
        raise ValueError(family)
    token_slots = torch.randint(
        0, SLOT_CAPACITY, (num_tokens,), generator=generator, dtype=torch.int32
    )
    return topk_ids.to(device), token_slots.to(device)


def _verdict(jit_samples: list[float], fused_samples: list[float]) -> str:
    """Decided only on unanimous sign with margin; otherwise tied."""
    decision = decide_cell(
        arm_a="jit", samples_a=jit_samples, arm_b="fused", samples_b=fused_samples
    )
    if decision.winner == "fused":
        return f"FUSED {decision.geo_a_over_b:.2f}x"
    if decision.winner == "jit":
        return f"JIT {1 / decision.geo_a_over_b:.2f}x"
    return f"tied ({decision.geo_a_over_b:.2f}x)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    suite = new_suite("align_boundary", source_revision=args.source_revision)

    # samples[(mode, V, P, family, arm)] -> list of medians (seed x repeat order)
    samples: dict[tuple, list[float]] = defaultdict(list)

    for v_total in V_GRID:
        lora_experts_per_adapter = v_total // SLOT_CAPACITY
        for num_pairs in P_GRID:
            num_tokens = num_pairs // TOP_K
            capacity = _routing_capacity(num_pairs, BLOCK_SIZE_M, v_total)
            families = ("iid", "hotset") if v_total in HOTSET_V else ("iid",)
            for family in families:
                for seed in SEEDS:
                    topk_ids, token_slots = _route(
                        family, lora_experts_per_adapter, num_tokens, seed, device
                    )
                    params = {
                        "V": v_total,
                        "P": num_pairs,
                        "family": family,
                        "seed": seed,
                        "lora_experts_per_adapter": lora_experts_per_adapter,
                        "slot_capacity": SLOT_CAPACITY,
                        "block_size": BLOCK_SIZE_M,
                    }

                    def jit_arm():
                        virtual_ids = _build_virtual_topk_ids(
                            topk_ids,
                            token_slots,
                            lora_experts_per_adapter,
                            SLOT_CAPACITY,
                            None,
                        )
                        return _align_block_size_jit(virtual_ids, BLOCK_SIZE_M, v_total)

                    def fused_arm():
                        return fused_align_block_size(
                            topk_ids,
                            token_slots,
                            lora_experts_per_adapter=lora_experts_per_adapter,
                            max_loras=SLOT_CAPACITY,
                            block_size=BLOCK_SIZE_M,
                            capacity=capacity,
                        )

                    arms = (("jit_route", jit_arm), ("fused_route", fused_arm))
                    for repeat in range(REPEATS):
                        # Alternate arm order so drift cannot favor one arm.
                        ordered = arms if repeat % 2 == 0 else arms[::-1]
                        for name, fn in ordered:
                            rec = measure(
                                fn,
                                suite=suite,
                                candidate=name,
                                boundary=BOUNDARY_ROUTE_INCLUSIVE,
                                params={**params, "repeat": repeat, "mode": "graph"},
                                graph_replay=True,
                            )
                            samples[("graph", v_total, num_pairs, family, name)].append(
                                rec.median_s * 1e6
                            )
                    # Eager: CPU launch cost included; reduced iterations
                    # (each is individually timed with an L2 flush, so eager
                    # iterations are expensive) but the SAME seed coverage as
                    # graph mode — the first version gated eager to one seed,
                    # which left every eager verdict on a 2-sample unanimity
                    # gate (satisfied by chance ~50% under a no-difference
                    # null; gate-2 review finding).
                    for repeat in range(REPEATS):
                        ordered = arms if repeat % 2 == 0 else arms[::-1]
                        for name, fn in ordered:
                            rec = measure(
                                fn,
                                suite=suite,
                                candidate=name,
                                boundary=BOUNDARY_ROUTE_INCLUSIVE,
                                params={
                                    **params,
                                    "repeat": repeat,
                                    "mode": "eager",
                                },
                                graph_replay=False,
                                warmup_iters=20,
                                replay_iters=200,
                            )
                            samples[("eager", v_total, num_pairs, family, name)].append(
                                rec.median_s * 1e6
                            )
            print(f"  V={v_total} P={num_pairs} done", flush=True)

    digest = write_suite(suite, args.output)

    print(
        f"\n{'V':>7}{'P':>7}{'family':>8}{'mode':>7}   "
        f"{'jit(us)':>17}{'fused(us)':>17}   verdict"
    )
    for mode in ("graph", "eager"):
        for v_total in V_GRID:
            for num_pairs in P_GRID:
                families = ("iid", "hotset") if v_total in HOTSET_V else ("iid",)
                for family in families:
                    jit = samples[(mode, v_total, num_pairs, family, "jit_route")]
                    fused = samples[(mode, v_total, num_pairs, family, "fused_route")]
                    if not jit or not fused:
                        continue
                    jit_span = f"{min(jit):.1f}-{max(jit):.1f}"
                    fused_span = f"{min(fused):.1f}-{max(fused):.1f}"
                    print(
                        f"{v_total:>7}{num_pairs:>7}{family:>8}{mode:>7}   "
                        f"{jit_span:>17}{fused_span:>17}   {_verdict(jit, fused)}"
                    )
    print(f"\n{len(suite.records)} records -> {args.output}  sha256={digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
