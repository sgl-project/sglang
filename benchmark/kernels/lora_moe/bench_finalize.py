"""Step-6 down-B + finalizer cells (plan §65.2): FM / FTOK / FSHARED / CUTE.

One timed thunk = everything a finalize schedule runs AFTER the down-A
bridge ``[P, R]`` exists; ``token_out`` is a NONZERO base destination
every arm accumulates into (bf16 plus a caller-selected fp32 sub-axis):

* ``fm`` — the 2-op materialized baseline: ``run_lora_b`` one-launch
  down-B writes the pair delta ``[P, H]``, then the route-free combine
  kernel folds it into the token output with the routed weights.
* ``ftok`` — the token-owned fused finalizer (``finalize_candidates``):
  ONE kernel, serial fixed-k order, routed weight exactly once, no pair
  delta ever materialized.
* ``fshared`` — the shared-B algebraic cut (shared cells only): weighted
  rank reduce ``[P,R] -> [T,R]`` folds the combine weights in rank space,
  then ONE grouped-by-adapter SGMV GEMM adds into the base.
* ``cute_tok`` / ``cute_shared`` — the CuTeDSL masked-grouped-GEMM arms
  (``finalize_cutedsl``), the mandatory §65.2 CUTE obligation; the
  token-width is their swept config axis.

OWNERSHIP x VALIDITY: per-expert cells run (fm, ftok, cute_tok) with
``b_down [L*E, H, R]``; shared cells run all five arms with the shared
``b_down`` viewed ``[L, H, R]`` (case signature ``shared_down_b``). Both
run on ``dense`` (ep_local ids, every routed pair owned) and ``ep8``
(global ids, 1/8 owned — rank-0 identity localization, the
bench_shared_down_b convention).

BOUNDARY HONESTY (mixed by design, mirroring bench_shared_down_b):

* ``fm`` on per-expert cells is PREPARED — its aligned plan is amortized
  from the down-A site, which needs the identical plan regardless.
* ``fm`` on shared cells is ROUTE_INCLUSIVE — the shared-outer plan is
  a per-layer build no other site pays for (bench_shared_down_b's
  ``one_launch_charged`` precedent), built in-thunk.
* ``ftok`` is PREPARED with a prebuilt RAW view — its honest route bill
  is zero device work (keys derive inline from the sources).
* ``fshared`` / ``cute_*`` are PREPARED at their most favorable
  metadata boundary (host-synced segments / plan builds outside timing).
  For ``fshared`` a loss there is a safe rejection; for ``cute_*`` a
  loss rejects only THIS staged composite (see the finalize_cutedsl
  STATUS note — the arms are controls, not optimized candidates).
  Winning re-opens the §64.12 charged segment/metadata builder question.

Destinations ACCUMULATE: timing replays re-accumulate garbage VALUES but
identical WORK (the sgmv_accum precedent); admission validates values
against a plain-torch fp32 oracle from a fresh base copy before any
timing record, and a numeric failure ABORTS (never a skip). Cells whose
route materializes zero valid pairs are skipped and recorded (the fp32
oracle is degenerate there).

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_finalize \
        --output finalize_v1.json --source-revision <sha> \
        [--ranks 16,64,128] [--validity dense,ep8]

SYNTHETIC ABI (14th review, P0 open — labeled, not yet implemented):
this bench measures a LoRA-only read-modify-write tail onto an ALREADY
FINALIZED token_out; production fuses base + LoRA in one provider pass
(routed_scale * sum_k w_k * (base_down[src2dst[k]] + delta_k)) and this
bench consumes neither base down_out, src2dst, nor
routed_scaling_factor. Suites publish as ``finalize_lora_tail_only_v1``
and every record carries ``abi: lora_tail_only`` — EXPLORATORY evidence
only, never qualification evidence.

ADJUDICATION POLICY (S5/6 review). These cells are SERIAL and ISOLATED,
so they measure one arm's critical path in isolation. That cannot
ELIMINATE an arm whose value is enabling OVERLAP:

* FM materializes the down-B result, letting down-B run OVERLAPPED
  with base W2 (the provider combine consumes both and waits for both);
* FTOK/FSHARED cut launches but can LENGTHEN the serial critical path.

Every family is therefore RETAINED through this bench. A loss here is
evidence about the serial path only; elimination requires the composed
leg measurement where the overlap either materializes or does not.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict

import torch

from benchmark.kernels.lora_moe.bench_common import (
    DECODE_T_MAX,
    padded_block_k_cap,
    regime_of,
    require_delta_close_chunked,
    require_writable_destination,
    skip_reason,
    write_skip_sidecar,
)
from benchmark.kernels.lora_moe.bench_lora_a import _LegFixture
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.finalize_candidates import (
    SHARED_RANK_REDUCE_DEFAULT_CONFIG,
    FinalizeExecutionSpec,
    build_shared_finalize_info,
    run_finalize,
)
from benchmark.kernels.lora_moe.finalize_cutedsl import (
    build_cutedsl_shared_finalize_plan,
    build_cutedsl_token_finalize_plan,
)
from benchmark.kernels.lora_moe.lora_a_candidates import run_lora_a
from benchmark.kernels.lora_moe.lora_a_cutedsl import (
    CutedslAConfig,
    supported_token_widths,
)
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.lora_b_candidates import run_lora_b
from benchmark.kernels.lora_moe.lora_b_execution import LoraBExecutionSpec
from benchmark.kernels.lora_moe.routes import generate_topk_weights
from benchmark.kernels.lora_moe.signal_gates import (
    BF16_RELATIVE_QUANTUM,
    MIN_SIGNAL_TO_NOISE,
    DegenerateSignalError,
    nan_poison_,
    require_delta_close,
)
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_PREPARED_INPUT,
    BOUNDARY_ROUTE_INCLUSIVE,
    measure,
    new_suite,
    write_suite,
)
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_ALIGNED,
    ROUTE_FUSED_IDS,
    ROUTE_RAW,
    build_virtual_expert_routing,
)

T_GRID = (4, 16, 64, 256, 2048, 8192)
SWEEP_T = {"decode_tiny": 4, "decode": 64, "prefill": 2048, "prefill_xl": 8192}
SEEDS = (11, 137, 997)
REPEATS = 2
VALIDITY_PRESETS = {"dense": (8, "ep_local"), "ep8": (8, "global")}
OWNERSHIPS = ("per_expert", "shared")
PER_EXPERT_ARMS = ("fm", "ftok", "cute_tok")
SHARED_ARMS = ("fm", "ftok", "fshared", "cute_tok", "cute_shared")
CUTE_ARMS = ("cute_tok", "cute_shared")
# Kernel launches of the ARM COMPOSITION itself (route builds excluded;
# `route_in_thunk` in params says who additionally builds in-thunk).
# fm = B GEMM + combine; fshared = reduce + SGMV; cute_tok = stage
# index_copy + GEMM + weighted scatter; cute_shared = reduce + stage +
# GEMM + scatter.
# cute staging is index_select FOLLOWED BY index_copy_ = TWO launches
# (13th S5/6 review: the prose was corrected but this record field was
# not) — cute_tok = gather + index_copy + GEMM + weighted scatter;
# cute_shared = reduce + gather + index_copy + GEMM + scatter.
ARM_LAUNCHES = {"fm": 2, "ftok": 1, "fshared": 2, "cute_tok": 4, "cute_shared": 5}

_SPEC_DOWN_A = LoraAExecutionSpec(site="down", ownership="grouped")
_SPEC_DOWN_B = LoraBExecutionSpec(
    site="down", ownership="grouped", slicing="one_launch_sliced"
)
_FINALIZE_SPECS = {
    ("fm", "per_expert"): FinalizeExecutionSpec(
        family="materialized", ownership="per_expert"
    ),
    ("fm", "shared"): FinalizeExecutionSpec(family="materialized", ownership="shared"),
    ("ftok", "per_expert"): FinalizeExecutionSpec(
        family="token_owned", ownership="per_expert"
    ),
    ("ftok", "shared"): FinalizeExecutionSpec(family="token_owned", ownership="shared"),
    ("fshared", "shared"): FinalizeExecutionSpec(
        family="shared_rank_reduce", ownership="shared"
    ),
}
_CANDIDATE_NAMES = {
    "fm": "finalize_materialized_2op",
    "ftok": "finalize_token_owned",
    "fshared": "finalize_shared_rank_reduce",
    "cute_tok": "finalize_cutedsl_token",
    "cute_shared": "finalize_cutedsl_shared",
}


def _arms(ownership: str) -> tuple[str, ...]:
    return SHARED_ARMS if ownership == "shared" else PER_EXPERT_ARMS


def _boundary(arm: str, ownership: str) -> str:
    if arm == "fm" and ownership == "shared":
        return BOUNDARY_ROUTE_INCLUSIVE
    return BOUNDARY_PREPARED_INPUT


def _candidate(arm: str, ownership: str, dest_dtype: torch.dtype) -> str:
    name = _CANDIDATE_NAMES[arm]
    if ownership == "shared" and arm in ("fm", "ftok", "cute_tok"):
        name += "_sharedB"
    if dest_dtype is torch.float32:
        name += "_fp32dest"
    return name


def _rank_tiles(rank: int) -> list[int]:
    """Legal rank-loop K tiles: padded powers of two through the rank cap.

    S5/6 review: the old form returned the RANK itself, so ranks 24/48/96
    emitted BLOCK_K values Triton cannot compile (tl.arange needs a
    power-of-two extent). Reuse Step 4's padded cap — every B kernel masks
    K against the true rank, so a padded power-of-two tile is exact.
    """
    cap = padded_block_k_cap(rank)
    return sorted({k for k in (16, 32, 64, 128) if k <= cap})


def _fm_grid(rank: int) -> list[dict]:
    return [
        {
            "b": {
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": group_m,
                "num_warps": b_warps,
                "num_stages": b_stages,
            },
            "combine": combine,
        }
        for bn in (64, 128, 256, 512)
        for group_m in (1, 4, 8, 16)
        for bk in _rank_tiles(rank)
        for b_warps in ((4,) if bn < 128 else (4, 8))
        for b_stages in ((2, 3) if bn < 256 else (2, 3, 4))
        for combine in (
            {"BLOCK_SIZE_T": 16, "BLOCK_SIZE_H": 128, "num_warps": 4, "num_stages": 2},
            {"BLOCK_SIZE_T": 32, "BLOCK_SIZE_H": 256, "num_warps": 8, "num_stages": 2},
        )
    ]


def _ftok_grid(rank: int) -> list[dict]:
    return [
        {
            "BLOCK_SIZE_H": bh,
            "BLOCK_SIZE_K": bk,
            "num_warps": warps,
            "num_stages": stages,
        }
        for bh in (64, 128, 256)
        for bk in _rank_tiles(rank)
        for warps in (4, 8)
        for stages in (2, 3)
    ]


def _fshared_gemm_grid(rank: int) -> list[dict]:
    """The SGMV GEMM section — Step 4's proven axes in full (15th review:
    BLOCK_S gains 32, BN gains 64; BS128 and small tiles were measured
    Step-4 winners)."""
    return [
        {
            "BLOCK_S": block_s,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": warps,
            "num_stages": stages,
        }
        for block_s in (16, 32, 64, 128, 256, 512)
        for block_n in (64, 128, 256, 512, 1024)
        for block_k in _rank_tiles(rank)
        for warps in (4, 8)
        for stages in (2, 3)
    ]


# 16th S5/6 review: BT 4/8 restore decode-tiny fairness (the sweep's
# smallest regime runs T=4, so BT16 processed 4x the needed token lanes)
# and warps 8 returns (the pre-split flat grid already tested it).
FSHARED_REDUCE_GRID = [
    {"BLOCK_SIZE_T": block_t, "num_warps": warps, "num_stages": stages}
    for block_t in (4, 8, 16, 32, 64)
    for warps in (2, 4, 8)
    for stages in (2, 3)
]


def _fshared_grid(rank: int) -> list[dict]:
    """PHASE 1 of the sectioned FSHARED sweep (15th review): the two
    kernels are sequential and independent, so composed time decomposes
    as t_reduce + t_gemm and their optima are separable — phase 1 sweeps
    the GEMM section at the default reduce, phase 2 (in the sweep loop)
    re-tunes the reduce section at the winning GEMM. This covers both
    kernels' full axes without the diagonal coupling a flat config
    forced, at grid cost |gemm| + |reduce| instead of their product."""
    reduce_default = dict(SHARED_RANK_REDUCE_DEFAULT_CONFIG["reduce"])
    return [
        {"reduce": reduce_default, "gemm": gemm} for gemm in _fshared_gemm_grid(rank)
    ]


def _arm_configs(arm: str, rank: int, cute_widths: tuple[int, ...]) -> list[dict]:
    if arm == "fm":
        return _fm_grid(rank)
    if arm == "ftok":
        return _ftok_grid(rank)
    if arm == "fshared":
        return _fshared_grid(rank)
    return [{"token_width": width} for width in cute_widths]


def _config_key(arm: str, config: dict) -> str:
    if arm == "fm":
        b, combine = config["b"], config["combine"]
        return (
            f"bn{b['BLOCK_SIZE_N']}-bk{b['BLOCK_SIZE_K']}"
            f"-g{b['GROUP_SIZE_M']}-w{b['num_warps']}-s{b['num_stages']}"
            f"+t{combine['BLOCK_SIZE_T']}-h{combine['BLOCK_SIZE_H']}"
            f"-w{combine['num_warps']}-s{combine['num_stages']}"
        )
    if arm == "fshared":
        reduce, gemm = config["reduce"], config["gemm"]
        return (
            f"r:t{reduce['BLOCK_SIZE_T']}-w{reduce['num_warps']}"
            f"-s{reduce['num_stages']}"
            f"+g:bs{gemm['BLOCK_S']}-bn{gemm['BLOCK_N']}-bk{gemm['BLOCK_K']}"
            f"-w{gemm['num_warps']}-s{gemm['num_stages']}"
        )
    if arm in CUTE_ARMS:
        return f"tw{config['token_width']}"
    if arm == "ftok":
        return (
            f"h{config['BLOCK_SIZE_H']}-bk{config['BLOCK_SIZE_K']}"
            f"-w{config['num_warps']}-s{config['num_stages']}"
        )
    return (
        f"t{config['BLOCK_SIZE_T']}-bs{config['BLOCK_S']}-bn{config['BLOCK_N']}"
        f"-bk{config['BLOCK_K']}-w{config['num_warps']}-s{config['num_stages']}"
    )


def _decided_pairs(ownership: str) -> list[tuple[str, str]]:
    """Canonical comparisons, emitted BY THE PRODUCER through decide_cell."""
    pairs = [("fm", arm) for arm in _arms(ownership)[1:]]
    pairs.append(("ftok", "cute_tok"))
    if ownership == "shared":
        pairs.append(("fshared", "cute_shared"))
        pairs.append(("ftok", "fshared"))
    # The fp32-destination sub-axis (§65.2 caller-selected destination).
    pairs.append(("ftok", "ftok_fp32"))
    if ownership == "shared":
        pairs.append(("fshared", "fshared_fp32"))
    return pairs


def _cutedsl_unavailable_reason(device: torch.device) -> str | None:
    major, _ = torch.cuda.get_device_capability(device)
    if major < 9:
        return f"sm{major}x < sm90"
    if importlib.util.find_spec("cutlass") is None:
        return "cutlass (CuTeDSL) not importable"
    return None


class _FinalizeFixture:
    """One (case, ownership) finalize workload: real bridge, oracle, buffers.

    The bridge is the ACTUAL grouped down-A output over the per-expert
    aligned plan (never random), its sentinel rows NaN-poisoned so any arm
    reading an invalid pair aborts at admission rather than passing on
    lucky garbage.  ``b_down`` comes pre-shaped by the case signature:
    ``[L*E, H, R]`` per-expert, ``[L, H, R]`` shared (``shared_down_b``).
    """

    def __init__(self, case, device, *, ownership: str, preset: str) -> None:
        self.case = case
        self.ownership = ownership
        self.preset = preset
        leg = _LegFixture(case, device)
        self.leg = leg
        self.num_tokens = case.num_tokens
        self.top_k = case.top_k
        self.hidden = case.moe_hidden_size
        self.rank = case.physical_rank
        self.b_down = leg.b_down
        # Real bridge: grouped down-A over the per-expert aligned plan (the
        # plan every arm's leg builds anyway for the A site — which is what
        # makes fm's per-expert reuse of it a PREPARED boundary).
        self.per_expert_aligned = leg.route(ROUTE_ALIGNED)
        run_lora_a(
            _SPEC_DOWN_A,
            input=leg.act_pair,
            weight=leg.a_down,
            output=leg.down_rank_out,
            routing=self.per_expert_aligned,
            config=PROVISIONAL_LAUNCH_CONFIG.lora_a,
        )
        self.bridge = leg.down_rank_out
        nan_poison_(self.bridge, (~leg.valid_pairs)[:, None].expand_as(self.bridge))
        self.pair_delta = leg.down_delta  # [P, H] bf16, fm's 2-op buffer
        self.combine_weights = generate_topk_weights(
            weight_distribution="seeded_random",
            num_tokens=case.num_tokens,
            top_k=case.top_k,
            seed=case.weight_seed,
        ).to(device)
        generator = torch.Generator(device="cpu").manual_seed(case.data_seed + 13)
        base = torch.randn(
            (case.num_tokens, self.hidden), generator=generator, dtype=torch.float32
        )
        if ownership == "shared":
            self.finalize_raw = self._shared_route(ROUTE_RAW)
            self.finalize_fused = self._shared_route(ROUTE_FUSED_IDS)
            self.finalize_info = build_shared_finalize_info(
                leg.token_slots, max_loras=case.slot_capacity, rank=self.rank
            )
            self.tok_bridge = torch.empty(
                (case.num_tokens, self.rank), dtype=torch.bfloat16, device=device
            )
        else:
            self.finalize_raw = leg.route(ROUTE_RAW)
            self.finalize_fused = leg.route(ROUTE_FUSED_IDS)
            self.finalize_info = None
            self.tok_bridge = None
        # Pair keys off the canonical fused view; cross-checked against the
        # host-derived mask so two independent derivations must agree.
        self.pair_veid = self.finalize_fused.virtual_topk_ids.view(-1).to(torch.int64)
        self.pair_valid = self.pair_veid >= 0
        if not torch.equal(self.pair_valid, leg.valid_pairs):
            raise RuntimeError(
                "fused-id validity disagrees with the host-derived pair mask "
                f"({ownership}, {preset}) — key derivations diverged"
            )
        self.has_valid_pairs = bool(self.pair_valid.any())
        self.oracle_delta = (
            self._fp32_token_delta()
            if self.has_valid_pairs
            else torch.zeros(
                (self.num_tokens, self.hidden), dtype=torch.float32, device=device
            )
        )
        # Accumulate-form arms write bf16(base + delta), so certifying
        # them needs the delta to clear the base's bf16 storage noise
        # (MIN_SIGNAL_TO_NOISE quanta of max|base|). The base is the
        # cell's only free scale — the bridge is the REAL down-A output —
        # so SHRINK it (never grow) until that rule holds with a 2x
        # margin; a unit-normal draw leaves a real-bridge delta at only
        # ~27 quanta and every cell skipped as degenerate. Kernel timing
        # is value-independent, so the workload is unchanged.
        signal = float(self.oracle_delta.abs().max())
        peak = float(base.abs().max())
        if signal > 0.0 and peak > 0.0:
            cap = signal / (2 * MIN_SIGNAL_TO_NOISE * BF16_RELATIVE_QUANTUM)
            if peak > cap:
                base = base * (cap / peak)
        self.base = {
            torch.bfloat16: base.to(torch.bfloat16).to(device),
            torch.float32: base.to(device),
        }
        self.token_out = {
            dtype: torch.empty_like(tensor) for dtype, tensor in self.base.items()
        }

    def _shared_route(self, view: str):
        # Global-domain ids (ep8) localize to identity on rank 0's owned
        # range, so the shared-outer bound doubles as the localization —
        # the bench_shared_down_b convention (Topology ep_rank defaults 0).
        return build_virtual_expert_routing(
            self.leg.topk_ids,
            self.leg.token_slots,
            lora_experts_per_adapter=1,
            max_loras=self.case.slot_capacity,
            block_size=self.case.routing_block_size,
            shared_outer_local_expert_count=self.case.num_experts_local,
            view=view,
        )

    def _fp32_token_delta(self) -> torch.Tensor:
        """Plain-torch fp32 group-loop oracle: ``sum_k w * (bridge @ B^T)``."""
        valid_rows = self.pair_valid.nonzero(as_tuple=False).view(-1)
        valid_veids = self.pair_veid[valid_rows]
        weights_flat = self.combine_weights.reshape(-1)
        delta = torch.zeros(
            (self.num_tokens, self.hidden),
            dtype=torch.float32,
            device=self.bridge.device,
        )
        for group in valid_veids.unique().tolist():
            rows = valid_rows[valid_veids == group]
            contribution = self.bridge[rows].float() @ self.b_down[group].float().t()
            contribution *= weights_flat[rows][:, None]
            delta.index_add_(0, rows // self.top_k, contribution)
        return delta

    def run(
        self,
        arm: str,
        config: dict,
        *,
        dest_dtype: torch.dtype,
        plans: dict | None = None,
    ) -> None:
        """One arm execution into the accumulating destination."""
        out = self.token_out[dest_dtype]
        if arm == "fm":
            routing = (
                self._shared_route(ROUTE_ALIGNED)  # per-layer build: charged
                if self.ownership == "shared"
                else self.per_expert_aligned
            )
            run_lora_b(
                _SPEC_DOWN_B,
                bridge=self.bridge,
                weight=self.b_down,
                destination=self.pair_delta,
                routing=routing,
                destination_offsets=(0,),
                config=config["b"],
            )
            run_finalize(
                _FINALIZE_SPECS[("fm", self.ownership)],
                routing=routing,
                combine_weights=self.combine_weights,
                token_out=out,
                config=config["combine"],
                pair_delta=self.pair_delta,
            )
        elif arm == "ftok":
            run_finalize(
                _FINALIZE_SPECS[("ftok", self.ownership)],
                routing=self.finalize_raw,
                combine_weights=self.combine_weights,
                token_out=out,
                config=config,
                bridge=self.bridge,
                b_down=self.b_down,
            )
        elif arm == "fshared":
            run_finalize(
                _FINALIZE_SPECS[("fshared", "shared")],
                routing=self.finalize_raw,
                combine_weights=self.combine_weights,
                token_out=out,
                config=config,
                bridge=self.bridge,
                b_down=self.b_down,
                finalize_info=self.finalize_info,
                tok_bridge=self.tok_bridge,  # preallocated: no in-thunk alloc
            )
        elif arm in CUTE_ARMS:
            plans[arm].run(
                bridge=self.bridge,
                combine_weights=self.combine_weights,
                token_out=out,
            )
        else:
            raise ValueError(f"unknown arm {arm!r}")


def _build_cute_plan(fixture: _FinalizeFixture, arm: str, token_width: int):
    """Build + verify + bind one CuTeDSL finalize plan (outside timing)."""
    config = CutedslAConfig(token_width=token_width)
    if arm == "cute_shared":
        plan = build_cutedsl_shared_finalize_plan(
            shared_route=fixture.finalize_fused,
            down_weight=fixture.b_down,
            config=config,
        )
    else:
        plan = build_cutedsl_token_finalize_plan(
            fused_route=fixture.finalize_fused,
            down_weight=fixture.b_down,
            config=config,
        )
    plan.build_metadata(verify=True)
    plan.require_binding(fixture.b_down, fixture.finalize_fused)
    return plan


def _admit(
    fixture: _FinalizeFixture,
    arm: str,
    config: dict,
    plans: dict | None,
    dest_dtype: torch.dtype,
    label: str,
) -> None:
    """fp32-oracle admission from a fresh base copy; failure ABORTS."""
    out = fixture.token_out[dest_dtype]
    base = fixture.base[dest_dtype]
    out.copy_(base)
    fixture.run(arm, config, dest_dtype=dest_dtype, plans=plans)
    torch.cuda.synchronize()
    if dest_dtype == torch.bfloat16:
        # Accumulate-form extraction: the arm WRITES bf16(base + delta),
        # so ``out - base`` carries up to one bf16 ulp of the BASE per
        # element and the fixed delta-domain rel-L2 gate becomes
        # unsatisfiable for a PERFECT kernel once L2(base)/L2(delta)
        # exceeds ~2.6 (first hit: dense/per_expert r16 decode_tiny,
        # rel_l2 1.685e-2 at max|err| = one base ulp). Gate with the
        # bounded accumulate-form allowance instead; cells where even
        # that is undecidable are skipped upstream by
        # ``_accumulate_cell_skip``.
        require_delta_close_chunked(
            out,
            fixture.oracle_delta,
            gate_dtype=torch.bfloat16,
            label=label,
            observed_base=base,
        )
    else:
        # FP32 destinations extract the delta exactly — keep the strict
        # delta-pure gate.
        require_delta_close(
            out.float() - base.float(),
            fixture.oracle_delta,
            gate_dtype=torch.bfloat16,
            label=label,
        )


def _accumulate_cell_skip(fixture: _FinalizeFixture) -> str | None:
    """Skip reason when the cell cannot certify bf16 accumulate-form arms.

    Probes the REAL admission gate with a perfect bf16 write of
    ``base + oracle_delta`` — the best any accumulate-form arm can do —
    so the certifiability rule can never drift from the gate itself.
    Degeneracy depends only on the fixture's base/delta norms (never on
    the arm or config), so one probe decides the whole cell. A plain
    gate FAILURE on the perfect write means the gate calibration itself
    is broken and propagates as the abort it is.
    """
    base = fixture.base[torch.bfloat16]
    perfect = (base.float() + fixture.oracle_delta).to(torch.bfloat16)
    try:
        require_delta_close_chunked(
            perfect,
            fixture.oracle_delta,
            gate_dtype=torch.bfloat16,
            label="accumulate-form cell probe",
            observed_base=base,
        )
    except DegenerateSignalError as error:
        return f"degenerate: {error}"
    return None


def _params(
    fixture: _FinalizeFixture,
    *,
    phase: str,
    regime: str,
    arm: str,
    config_key: str,
    dest_dtype: torch.dtype,
    seed: int | None = None,
    repeat: int | None = None,
) -> dict:
    pair_delta_bytes = fixture.pair_delta.numel() * 2
    return {
        # 14th S5/6 review: the finalizer ABI gap, stamped on EVERY record
        # — this bench measures a LoRA-only tail onto an already-finalized
        # output; it does not consume base down_out / src2dst /
        # routed_scaling_factor. Not qualification evidence.
        "abi": "lora_tail_only",
        "case_id": fixture.case.case_id,
        "phase": phase,
        "T": fixture.case.num_tokens,
        "P": fixture.leg.num_pairs,
        "rank": fixture.rank,
        "validity": fixture.preset,
        "ownership": fixture.ownership,
        "regime": regime,
        "arm": arm,
        "config": config_key,
        "dest_dtype": "float32" if dest_dtype is torch.float32 else "bfloat16",
        "arm_launches": ARM_LAUNCHES[arm],
        "route_in_thunk": arm == "fm" and fixture.ownership == "shared",
        # §65.2 evidence: the 2-op path writes AND re-reads the pair delta;
        # every fused arm eliminates both movements.
        "pair_delta_buffer_bytes": pair_delta_bytes,
        "pair_delta_bytes_eliminated": 0 if arm == "fm" else 2 * pair_delta_bytes,
        "seed": seed,
        "repeat": repeat,
    }


def _build_bench_case(
    device,
    *,
    preset: str,
    ownership: str,
    num_tokens: int,
    rank: int,
    seed: int,
    source_revision: str,
):
    ep_size, domain = VALIDITY_PRESETS[preset]
    return build_case(
        device=str(device),
        model_preset="qwen35_35b",
        topology=Topology(tp_size=8, ep_size=ep_size),
        adapter_cell=AdapterCell(
            active_adapters=4, include_base_rows=True, slot_capacity=8
        ),
        route_generator="iid",
        expert_id_domain=domain,
        num_tokens=num_tokens,
        active_rank=rank,
        shared_factor_signature=(
            "shared_down_b" if ownership == "shared" else "per_expert"
        ),
        seed=seed,
        source_revision=source_revision,
    )


def select_between_phases(
    phase1: tuple[dict, float], phase2: tuple[dict | None, float]
) -> tuple[dict, float]:
    """Pick the faster of the two FSHARED sweep phases (18th review).

    Phase 2 excludes the default reduce as DEDUPLICATION (phase 1 already
    timed it), never as a forfeit: if every alternative is slower — or
    all were skipped (phase2 config None) — the phase-1 winner stands.
    """
    config, median = phase1
    phase2_config, phase2_median = phase2
    if phase2_config is not None and phase2_median < median:
        return phase2_config, phase2_median
    return config, median


def _sweep_arm(
    suite,
    fixture: _FinalizeFixture,
    *,
    regime: str,
    arm: str,
    cute_widths: tuple[int, ...],
    cell: str,
    skips: list[dict],
    configs: list[dict] | None = None,
) -> tuple[dict | None, float]:
    """Tune one arm on one cell; returns (best_config, best_median).

    ``configs`` overrides the arm's default grid — used by FSHARED's
    phase-2 reduce re-tune (15th review).
    """
    best_config, best_median = None, float("inf")
    for config in (
        configs if configs is not None else _arm_configs(arm, fixture.rank, cute_widths)
    ):
        key = _config_key(arm, config)
        # S5/6 verification, MAJOR: build failures were demoted to sidecar
        # skips, letting the MANDATORY cute arms vanish silently.
        # Capability is pre-gated at startup; past it, failures abort —
        # EXCEPT the schedule builder's per-config geometric refusal
        # (cluster packing overflow at a narrow token width), which is an
        # infeasible tile like any Triton resource skip: the arm survives
        # through its wider widths and the skip is recorded (first hit:
        # T=8192 needs 1639 clusters at width 8, packing holds 1024).
        plans = None
        if arm in CUTE_ARMS:
            try:
                plans = {arm: _build_cute_plan(fixture, arm, config["token_width"])}
            except ValueError as error:
                if "token clusters; the packing holds" not in str(error):
                    raise
                skips.append(
                    {
                        "arm": arm,
                        "cell": cell,
                        "config": key,
                        "reason": f"infeasible tile: {error}",
                    }
                )
                continue
        try:
            _admit(
                fixture,
                arm,
                config,
                plans,
                torch.bfloat16,
                f"sweep {arm} {key} {cell}",
            )
        except Exception as error:
            reason = skip_reason(error)
            if reason is None:
                raise
            skips.append({"arm": arm, "cell": cell, "config": key, "reason": reason})
            continue
        record = measure(
            lambda c=config, p=plans: fixture.run(
                arm, c, dest_dtype=torch.bfloat16, plans=p
            ),
            suite=suite,
            candidate=f"sweep_{_candidate(arm, fixture.ownership, torch.bfloat16)}",
            boundary=_boundary(arm, fixture.ownership),
            params=_params(
                fixture,
                phase="sweep",
                regime=regime,
                arm=arm,
                config_key=key,
                dest_dtype=torch.bfloat16,
            ),
            graph_replay=True,
            warmup_iters=10,
            replay_iters=100,
        )
        if record.median_s < best_median:
            best_config, best_median = config, record.median_s
    return best_config, best_median


def _run_sweeps(suite, device, presets, ranks, cute_widths, skips) -> dict:
    """4-regime per-(validity, ownership, rank) tuning; returns the table."""
    best: dict = {}
    for preset in presets:
        for ownership in OWNERSHIPS:
            for rank in ranks:
                for regime, num_tokens in SWEEP_T.items():
                    case = _build_bench_case(
                        device,
                        preset=preset,
                        ownership=ownership,
                        num_tokens=num_tokens,
                        rank=rank,
                        seed=SEEDS[0],
                        source_revision=suite.source_revision,
                    )
                    fixture = _FinalizeFixture(
                        case, device, ownership=ownership, preset=preset
                    )
                    cell = f"{preset}/{ownership} r{rank} {regime}(T={num_tokens})"
                    if not fixture.has_valid_pairs:
                        skips.append(
                            {"cell": cell, "reason": "degenerate: zero valid pairs"}
                        )
                        print(f"SWEEP {cell}: SKIPPED (zero valid pairs)", flush=True)
                        continue
                    reason = _accumulate_cell_skip(fixture)
                    if reason is not None:
                        skips.append({"cell": cell, "reason": reason})
                        print(f"SWEEP {cell}: SKIPPED ({reason})", flush=True)
                        continue
                    for arm in _arms(ownership):
                        config, median = _sweep_arm(
                            suite,
                            fixture,
                            regime=regime,
                            arm=arm,
                            cute_widths=cute_widths,
                            cell=cell,
                            skips=skips,
                        )
                        if config is None:
                            if arm in CUTE_ARMS:
                                print(
                                    f"SWEEP {cell} [{arm}]: no admissible build "
                                    "(recorded skip)",
                                    flush=True,
                                )
                                continue
                            raise RuntimeError(f"no admissible {arm} config at {cell}")
                        if arm == "fshared":
                            # PHASE 2 (15th review): re-tune the reduce
                            # section at the winning GEMM — the kernels are
                            # sequential and independent, so the composed
                            # optimum decomposes (see _fshared_grid).
                            # 18th review: phase 2 excludes the default
                            # reduce (already timed in phase 1) but the
                            # phases COMPETE on median — the exclusion is
                            # deduplication, never a forfeit; a default
                            # reduce faster than all 29 alternatives keeps
                            # the win.
                            phase2_config, phase2_median = _sweep_arm(
                                suite,
                                fixture,
                                regime=regime,
                                arm=arm,
                                cute_widths=cute_widths,
                                cell=cell,
                                skips=skips,
                                configs=[
                                    {"reduce": reduce, "gemm": config["gemm"]}
                                    for reduce in FSHARED_REDUCE_GRID
                                    if reduce != config["reduce"]
                                ],
                            )
                            config, median = select_between_phases(
                                (config, median),
                                (phase2_config, phase2_median),
                            )
                        best[(preset, ownership, rank, regime, arm)] = config
                        print(
                            f"SWEEP {cell} [{arm}]: {_config_key(arm, config)} "
                            f"({median * 1e6:.1f}us)",
                            flush=True,
                        )
    return best


def _decided_entries(
    fixture: _FinalizeFixture,
    arms: list[str],
    tuned: dict,
    skips: list[dict],
    label: str,
) -> tuple[list[tuple], dict] | None:
    """(sample_key, arm, dest_dtype, config) rows + cute plans for one seed.

    Returns None when a cute plan build fails — the whole seed is dropped
    so every arm keeps PAIRED sample lists (decide_cell's precondition).
    """
    # S5/6 verification, MAJOR: a cute build failure here used to drop the
    # WHOLE seed for ALL arms (return None), degrading even the Triton
    # arms' sample counts. Past the startup capability gate, build
    # failures are bugs and abort the run.
    plans: dict = {}
    for arm in arms:
        if arm in CUTE_ARMS:
            plans[arm] = _build_cute_plan(fixture, arm, tuned[arm]["token_width"])
    entries = [(arm, arm, torch.bfloat16, tuned[arm]) for arm in arms]
    for arm in ("ftok", "fshared"):
        if arm in arms:
            entries.append((f"{arm}_fp32", arm, torch.float32, tuned[arm]))
    return entries, plans


def _run_decided(suite, device, presets, ranks, best, skips) -> dict:
    """Seeded interleaved decided cells at the tuned configs."""
    samples: dict = defaultdict(lambda: defaultdict(list))
    for preset in presets:
        for ownership in OWNERSHIPS:
            for rank in ranks:
                for num_tokens in T_GRID:
                    regime = regime_of(num_tokens)
                    arms = [
                        arm
                        for arm in _arms(ownership)
                        if (preset, ownership, rank, regime, arm) in best
                    ]
                    if not arms:
                        continue
                    tuned = {
                        arm: best[(preset, ownership, rank, regime, arm)]
                        for arm in arms
                    }
                    modes = (True,) if num_tokens <= DECODE_T_MAX else (True, False)
                    for seed in SEEDS:
                        case = _build_bench_case(
                            device,
                            preset=preset,
                            ownership=ownership,
                            num_tokens=num_tokens,
                            rank=rank,
                            seed=seed,
                            source_revision=suite.source_revision,
                        )
                        fixture = _FinalizeFixture(
                            case, device, ownership=ownership, preset=preset
                        )
                        label = f"{preset}/{ownership} r{rank} T={num_tokens} s{seed}"
                        if not fixture.has_valid_pairs:
                            skips.append(
                                {
                                    "cell": label,
                                    "reason": "degenerate: zero valid pairs",
                                }
                            )
                            continue
                        reason = _accumulate_cell_skip(fixture)
                        if reason is not None:
                            skips.append({"cell": label, "reason": reason})
                            continue
                        built = _decided_entries(fixture, arms, tuned, skips, label)
                        if built is None:
                            continue
                        entries, plans = built
                        for key, arm, dest_dtype, config in entries:
                            _admit(
                                fixture,
                                arm,
                                config,
                                plans,
                                dest_dtype,
                                f"decided {key} {label}",
                            )
                        cell = (preset, ownership, rank, num_tokens)
                        for graph in modes:
                            mode = "graph" if graph else "eager"
                            for repeat in range(REPEATS):
                                ordered = entries if repeat % 2 == 0 else entries[::-1]
                                for key, arm, dest_dtype, config in ordered:
                                    record = measure(
                                        lambda a=arm, c=config, d=dest_dtype: (
                                            fixture.run(a, c, dest_dtype=d, plans=plans)
                                        ),
                                        suite=suite,
                                        candidate=_candidate(
                                            arm, ownership, dest_dtype
                                        ),
                                        boundary=_boundary(arm, ownership),
                                        params=_params(
                                            fixture,
                                            phase="decided",
                                            regime=regime,
                                            arm=arm,
                                            config_key=_config_key(arm, config),
                                            dest_dtype=dest_dtype,
                                            seed=seed,
                                            repeat=repeat,
                                        ),
                                        graph_replay=graph,
                                    )
                                    samples[(*cell, mode)][key].append(record.median_s)
                        print(f"decided {label}: {len(entries)} arms", flush=True)
    return samples


def _print_decisions(samples: dict) -> None:
    for cell in sorted(samples):
        preset, ownership, rank, num_tokens, mode = cell
        for arm_a, arm_b in _decided_pairs(ownership):
            samples_a = samples[cell].get(arm_a)
            samples_b = samples[cell].get(arm_b)
            if not samples_a or not samples_b or len(samples_a) != len(samples_b):
                # S5/6 verification (m14): silence here hid missing arms —
                # an auditor could not tell "not adjudicated" from "tied".
                print(
                    f"{mode:5s} {preset:5s} {ownership:10s} r{rank:<4d} "
                    f"T={num_tokens:<5d} {arm_a}/{arm_b:18s} UNPAIRED "
                    f"({len(samples_a or [])}/{len(samples_b or [])} samples)"
                )
                continue
            decision = decide_cell(
                arm_a=arm_a,
                samples_a=samples_a,
                arm_b=arm_b,
                samples_b=samples_b,
            )
            print(
                f"{mode:5s} {preset:5s} {ownership:10s} r{rank:<4d} "
                f"T={num_tokens:<5d} {arm_a}/{arm_b:16s} "
                f"geo(a/b)={decision.geo_a_over_b:.3f} -> "
                f"{decision.winner or 'tied'}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,64,128")
    parser.add_argument("--validity", default="dense,ep8")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    require_writable_destination(arguments.output)
    # S5/6 verification (m11): validate the rank axis BEFORE CUDA init,
    # like bench_fused_middle — an unsupported rank used to surface only
    # mid-run inside a kernel launcher.
    ranks = tuple(int(value) for value in arguments.ranks.split(","))
    for rank in ranks:
        if not 16 <= rank <= 128 or rank % 8:
            raise ValueError(
                f"rank {rank} outside the supported 16..128 multiple-of-8 "
                "range (CuTeDSL TMA alignment + register-resident premise)"
            )
    torch.cuda.set_device(device)
    ranks = tuple(int(rank) for rank in arguments.ranks.split(","))
    presets = tuple(arguments.validity.split(","))
    for preset in presets:
        if preset not in VALIDITY_PRESETS:
            raise ValueError(
                f"unknown validity preset {preset!r}; expected one of "
                f"{tuple(VALIDITY_PRESETS)}"
            )
    # 14th S5/6 review: the ABI gap is LABELED into the artifact identity
    # so these suites can never be mistaken for qualification evidence.
    suite = new_suite(
        "finalize_lora_tail_only_v1", source_revision=arguments.source_revision
    )
    skips: list[dict] = []

    unavailable = _cutedsl_unavailable_reason(device)
    if unavailable is None:
        cute_widths = supported_token_widths(device)
    else:
        cute_widths = ()
        skips.append({"arm": "cute_*", "cell": "ALL", "reason": unavailable})
        print(f"CuTeDSL arms disabled: {unavailable}", flush=True)

    best = _run_sweeps(suite, device, presets, ranks, cute_widths, skips)
    samples = _run_decided(suite, device, presets, ranks, best, skips)
    _print_decisions(samples)

    write_skip_sidecar(arguments.output, skips)
    if unavailable is None and not any(
        "cute" in record.candidate for record in suite.records
    ):
        raise RuntimeError(
            "CuTeDSL was enabled but produced ZERO records; refusing to "
            "publish Triton-only evidence for a mandatory arm"
        )
    digest = write_suite(suite, arguments.output)
    print(f"{len(suite.records)} records -> {arguments.output} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
