"""Step-5 fused-middle bench: M vs BA/AD/FULL/CUTE (plan §65.1).

The MIDDLE of one MoE LoRA leg is gate/up-B -> activation join -> down-A.
Every arm consumes the same inputs (``bridge_gu`` [P, S*R], ``base_gu``
[P, S*W]) and produces the same common output boundary (``act`` [P, W] —
also the base W2 input, so it stays materialized — plus ``down_rank_out``
[P, R]):

* **m** (baseline): one-launch B -> join kernel -> grouped down-A
  (3 launches; ``gate_up_delta`` [P, S*W] round-trips HBM and ``act`` is
  re-read by the separate down-A).
* **b_act**: fused B+activation + grouped down-A (2 launches; kills the
  delta buffer).
* **act_down_a**: one-launch B + fused activation+down-A (2 launches;
  kills the act re-read; deterministic serial W-tile loop).
* **full**: everything in ONE kernel (kills delta AND the act re-read).
* **cutedsl**: the masked-grouped-GEMM composite
  (``fused_middle_cutedsl``): stage -> slice GEMM -> join -> down GEMM ->
  scatter (5 launches; pair-domain delta eliminated, staged buffers added
  — recorded in params, not netted away).

BOUNDARY HONESTY: the Triton arms and the CUTE arm are decided at
``BOUNDARY_PREPARED_INPUT`` — the aligned plan is prebuilt because the
leg's A and B stages already require it, and the CUTE plan's dispatch +
schedules are prebuilt because that is ITS most favorable metadata
boundary (14th review: losing there rejects only THIS STAGED
IMPLEMENTATION — see the STATUS note in fused_middle_cutedsl and the
adjudication policy below, which retains every family; winning re-opens
the §64.12 charged-builder question).  The ``full_charged``/``cutedsl_charged``
pair replays the comparison at ``BOUNDARY_ROUTE_INCLUSIVE`` (aligned plan
rebuilt in-thunk vs ``build_metadata(verify=False)`` in-thunk) so the
charged verdict is measured, not argued.

Discipline (bench_shared_down_b v5 lineage): per-(geometry, validity,
rank, regime) config sweeps with fail-closed skips persisted to a
sidecar; FP32-oracle admission (``reference_fused_middle``) before any
timing record; seeded decided phase (3 seeds x 2 interleaved repeats) at
the full T grid; canonical pairs emitted through ``decide_cell`` by this
producer.  The non-gated ReLU^2 guardrail runs the same machinery as one
decided sub-grid on ``nemotron3_super`` (``dense`` validity only).

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_fused_middle \
        --output fm1.json --source-revision <sha> \
        [--ranks 16,64,128] [--validity dense,ep8] \
        [--geometries gated,relu2] [--cutedsl-token-widths auto]

SYNTHETIC ABI (14th review, P0 open — labeled, not yet implemented):
every tensor here is PAIR-MAJOR synthetic ([P, 2W] base, [P, W] act);
the qualified BF16 providers expose masked [E, m_max, *] buffers linked
by ws.src2dst with gate_first/interleaved layout flags. Suites publish
as ``fused_middle_synthetic_pair_abi_v1`` and every record carries
``abi: synthetic_pair_major`` — EXPLORATORY evidence for ranking fusion
ideas, NOT qualification evidence for the backend.

KNOWN COMPOSITION GAP (S5/6 review, unresolved): the Triton middle
kernels require a PAIR-MAJOR [P, 2R] bridge, but the selected
shared-outer gate-A path produces a token-deduplicated [T, 2R] bridge.
The CuTeDSL composite supports that axis; this bench never enables it.
Until the corrected provider-domain ABI lands, S5 results do not compose
with the token-dedup shared-A winner.

ADJUDICATION POLICY (S5/6 review). These cells are SERIAL and ISOLATED,
so they measure one arm's critical path in isolation. That cannot
ELIMINATE an arm whose value is enabling OVERLAP:

* BA releases the activation early, letting base W2 overlap a
  standalone down-A/down-B;
* M (the materialized join) keeps every stage separable, the maximum
  overlap surface;
* AD/FULL cut launches but can LENGTHEN the serial critical path.

Every family is therefore RETAINED through this bench. A loss here is
evidence about the serial path only; elimination requires the composed
leg measurement where the overlap either materializes or does not.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import triton
import triton.language as tl

from benchmark.kernels.lora_moe.bench_common import (
    DECODE_T_MAX,
    exhaustive_grouped_lora_b_grid,
    padded_block_k_cap,
    regime_of,
    require_writable_destination,
    skip_reason,
    write_skip_sidecar,
)
from benchmark.kernels.lora_moe.bench_lora_a import _LegFixture
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.fused_middle_candidates import (
    FusedMiddleSpec,
    run_fused_middle,
)
from benchmark.kernels.lora_moe.fused_middle_cutedsl import (
    CutedslAConfig,
    build_cutedsl_fused_middle_plan,
    invoke_cutedsl_fused_middle,
    reference_fused_middle,
    supported_token_widths,
)
from benchmark.kernels.lora_moe.lora_a_candidates import run_lora_a
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.lora_b_candidates import run_lora_b
from benchmark.kernels.lora_moe.lora_b_execution import LoraBExecutionSpec
from benchmark.kernels.lora_moe.signal_gates import (
    nan_poison_,
    require_delta_close,
    require_finite,
)
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ISOLATED,
    BOUNDARY_PREPARED_INPUT,
    BOUNDARY_ROUTE_INCLUSIVE,
    measure,
    new_suite,
    write_suite,
)
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import ROUTE_ALIGNED, ROUTE_FUSED_IDS

T_GRID = (4, 16, 64, 256, 2048, 8192)
SWEEP_T = {"decode_tiny": 4, "decode": 64, "prefill": 2048, "prefill_xl": 8192}
SEEDS = (11, 137, 997)
REPEATS = 2
VALIDITY_PRESETS = {"dense": (8, "ep_local"), "ep8": (8, "global")}
# Primary anchor geometry + the §65.1 non-gated ReLU^2 guardrail sub-grid.
GEOMETRY_PRESETS = {"gated": "qwen35_35b", "relu2": "nemotron3_super"}
GEOMETRY_VALIDITY = {"gated": ("dense", "ep8"), "relu2": ("dense",)}
# cases.py names the gated activation "silu_glu"; the fused-middle
# kernels call the same semantics "silu_mul" — the documented mapping.
_SPEC_ACTIVATION = {"silu_glu": "silu_mul", "relu2": "relu2"}

# Decided arm order (interleaved forward/backward per repeat).  The
# charged pair replays FULL-vs-CUTE at the route-inclusive boundary.
# 14th S5/6 review: the materialized join was pinned at BLOCK_W=512 /
# 4 warps — a weak control exaggerates fusion wins. Small targeted sweep.
JOIN_DEFAULT_CONFIG = {"BLOCK_W": 512, "num_warps": 4}


# 15th S5/6 review: the decided phase silently DISCARDED the tuned join
# (tuned.get("m_join") -> None -> default), so the join sweep had no
# effect on the M-vs-fused verdict. One constant now drives both sweep
# storage and decided loading, and run_arm reads the key HARD.
def load_decided_tuned(best: dict, cell_key: tuple) -> dict:
    """The ONE loader carrying sweep winners into the decided phase
    (16th review: shared by production and the regression test)."""
    return {family: best[(*cell_key, family)] for family in DECIDED_TUNED_FAMILIES}


DECIDED_TUNED_FAMILIES = (
    "m_b",
    "m_join",
    "m_down",
    "b_act",
    "act_down_a",
    "full",
)
JOIN_GRID = [
    {"BLOCK_W": bw, "num_warps": warps}
    for bw in (128, 256, 512, 1024)
    for warps in (4, 8)
]
ARMS = ("m", "b_act", "act_down_a", "full", "cutedsl")
CHARGED_ARMS = ("full_charged", "cutedsl_charged")
BOUNDARIES = {
    "m": BOUNDARY_PREPARED_INPUT,
    "b_act": BOUNDARY_PREPARED_INPUT,
    "act_down_a": BOUNDARY_PREPARED_INPUT,
    "full": BOUNDARY_PREPARED_INPUT,
    "cutedsl": BOUNDARY_PREPARED_INPUT,
    "full_charged": BOUNDARY_ROUTE_INCLUSIVE,
    "cutedsl_charged": BOUNDARY_ROUTE_INCLUSIVE,
}
DECIDED_PAIRS = (
    ("m", "b_act"),
    ("m", "act_down_a"),
    ("m", "full"),
    ("m", "cutedsl"),
    ("full", "b_act"),
    ("full", "act_down_a"),
    ("full", "cutedsl"),
    ("full_charged", "cutedsl_charged"),
)
# AD/FULL write down_rank_out universally (sentinel rows EXACT ZERO);
# m/b_act finish with grouped down-A, which preserves sentinel rows
# (production semantics) — their admission compares valid rows only.
_UNIVERSAL_DOWN_ARMS = frozenset(
    ("act_down_a", "full", "full_charged", "cutedsl", "cutedsl_charged")
)
_LAUNCHES = {
    "m": 3,
    "b_act": 2,
    "act_down_a": 2,
    "full": 1,
    "full_charged": 1,
    "cutedsl": 5,
    "cutedsl_charged": 5,
}

_SPEC_B_STOCK = LoraBExecutionSpec(site="gate_up", ownership="grouped")
_SPEC_B_ONE_LAUNCH = LoraBExecutionSpec(
    site="gate_up", ownership="grouped", slicing="one_launch_sliced"
)
_SPEC_DOWN_GROUPED = LoraAExecutionSpec(site="down", ownership="grouped")


_KEY_SHORT = (
    ("BLOCK_W", "jw"),
    ("BLOCK_SIZE_W", "bw"),
    ("BLOCK_SIZE_N", "bn"),
    ("BLOCK_SIZE_K", "bk"),
    ("BLOCK_SIZE_M", "m"),
    ("GROUP_SIZE_M", "g"),
    ("num_warps", "w"),
    ("num_stages", "s"),
    ("token_width", "tw"),
)


def _cfg_key(config: dict) -> str:
    return "-".join(
        f"{short}{config[name]}" for name, short in _KEY_SHORT if name in config
    )


def _middle_grid(rank: int, *, rank_loop: bool, swizzle: bool = False) -> list[dict]:
    """BA/AD/FULL sweep grid. AD has no rank loop; its BLOCK_SIZE_K is a
    fixed placeholder (accepted-but-ignored, config-grid uniformity).

    14th/15th S5/6 review: BA uses the GROUP_SIZE_M program swizzle
    (b_act grid) so it sweeps the axis; FULL launches a 1-D serial grid
    and AD one program per m-block — both IGNORE it, so sweeping it
    there would repeat every effective config four times and record a
    misleading parameter. The axis is swept exactly where consumed.

    17th S5/6 review: BA (swizzle=True) also gets the WIDE axes its
    materialized control searches — BLOCK_SIZE_W through 512 and stage 4.
    Evidence, not speculation: 18/32 Step-4 gate/up one-launch winners
    sat at BN256/512 and 4/32 at stage 4, and the sliced B kernel's
    BLOCK_SIZE_N tiles the same gate/up slice BA's BLOCK_SIZE_W does.
    AD/FULL stay constrained: their LIVE down-rank accumulator
    ([BLOCK_M, R2] fp32 held across the W loop) is a materially
    different register regime; resource-heavy BA points follow the
    existing skip path.
    """
    # 16th S5/6 review: padded K tiles, same cap as the materialized
    # control — rank 48 was missing BK64 and rank 96 BK128, though the
    # kernels mask K and one padded dot can replace two K-loop dots.
    rank_ks = tuple(k for k in (16, 32, 64, 128) if k <= padded_block_k_cap(rank))
    return [
        {
            "BLOCK_SIZE_W": bw,
            "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": group_m,
            "num_warps": warps,
            "num_stages": stages,
        }
        for bw in ((32, 64, 128, 256, 512) if swizzle else (32, 64, 128))
        for bk in (rank_ks if rank_loop else rank_ks[:1])
        for group_m in ((1, 4, 8, 16) if swizzle else (8,))
        for warps in (4, 8)
        for stages in ((2, 3, 4) if swizzle else (2, 3))
    ]


def _b_grid(rank: int) -> list[dict]:
    """The MATERIALIZED control's grid — Step 4's axes, not a subset.

    S5/6 review: this stopped at BN=128 with GROUP_SIZE_M pinned to 8,
    while Step 4 found real grouped-B winners at BN 256/512 and other
    GROUP_SIZE_M values. Eliminating arm M against an under-tuned control
    would be the same mistake the pruned-tuner replay caught in Step 4.
    """
    return list(exhaustive_grouped_lora_b_grid(rank=rank, stock=False))


def _down_grid(rank: int, width: int) -> list[dict]:
    return [
        {
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": 8,
            "num_warps": warps,
            "num_stages": stages,
        }
        for bn in (16, 32, 64, 128)
        if bn <= max(rank, 16)
        for bk in (32, 64, 128)
        if bk <= width
        for warps in (4, 8)
        for stages in (2, 3, 4)
    ]


@triton.jit
def _materialized_join_kernel(
    delta_ptr,  # [P, S*W] bf16 (B's zero-fill contract covers sentinels)
    base_ptr,  # [P, S*W] bf16
    act_ptr,  # [P, W] bf16 out
    inter,
    stride_dm,
    stride_pm,
    stride_am,
    NUM_SLICES: tl.constexpr,
    ACT_RELU2: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Arm M's join: act = activation(base + materialized delta), one
    launch, one program per pair (the ``_join_staged_rows_kernel`` shape).
    Rows must be element-stride-1; only row strides are parameterized.
    Sentinel pairs are covered by B's exact-zero delta fill, so ``act`` is
    universal here too."""
    pair = tl.program_id(0).to(tl.int64)
    vec = tl.arange(0, BLOCK_W).to(tl.int64)
    for start in tl.range(0, inter, BLOCK_W):
        offs = start + vec
        w_mask = offs < inter
        gate = tl.load(base_ptr + pair * stride_pm + offs, mask=w_mask, other=0.0).to(
            tl.float32
        )
        gate += tl.load(delta_ptr + pair * stride_dm + offs, mask=w_mask, other=0.0).to(
            tl.float32
        )
        if NUM_SLICES == 2:
            up = tl.load(
                base_ptr + pair * stride_pm + inter + offs, mask=w_mask, other=0.0
            ).to(tl.float32)
            up += tl.load(
                delta_ptr + pair * stride_dm + inter + offs, mask=w_mask, other=0.0
            ).to(tl.float32)
        else:
            up = gate
        if ACT_RELU2:
            rectified = tl.maximum(gate, 0.0)
            act = rectified * rectified
        else:
            act = gate * tl.sigmoid(gate) * up
        tl.store(
            act_ptr + pair * stride_am + offs,
            act.to(act_ptr.dtype.element_ty),
            mask=w_mask,
        )


class _MiddleFixture:
    """One case's middle-segment tensors, reusing the ``_LegFixture``.

    ``gate_rank_out`` is seeded as the bridge (its sentinel rows are
    NaN-poisoned — every arm's contract says they are never read, so any
    leak fails ``require_finite`` at admission and is visible in every
    timed replay too); ``gate_up_delta`` / ``act_pair`` / ``down_rank_out``
    are the delta / act / rank-out buffers; ``base_gu`` is synthesized
    seeded randn (the base W13 output the join adds).  For the non-gated
    geometry the 2-slice leg buffers are column-sliced to 1-slice shapes
    (strided views; every consumer takes row strides).
    """

    def __init__(self, case, device) -> None:
        self.leg = _LegFixture(case, device)
        self.case = case
        self.activation = _SPEC_ACTIVATION[case.activation]
        self.num_slices = 2 if self.activation == "silu_mul" else 1
        slices, rank, width = self.num_slices, case.physical_rank, self.leg.intermediate
        self.rank, self.width = rank, width
        generator = torch.Generator(device="cpu").manual_seed(case.data_seed + 11)
        self.leg.gate_rank_out.copy_(
            (torch.randn((self.leg.num_pairs, 2 * rank), generator=generator) * 0.5)
            .to(torch.bfloat16)
            .to(device)
        )
        self.bridge = self.leg.gate_rank_out[:, : slices * rank]
        self.valid = self.leg.valid_pairs
        self.has_valid = bool(self.valid.any().item())
        nan_poison_(self.bridge, mask=~self.valid[:, None])
        self.base_gu = (
            (
                torch.randn((self.leg.num_pairs, slices * width), generator=generator)
                * 0.5
            )
            .to(torch.bfloat16)
            .to(device)
        )
        self.gate_up_delta = self.leg.gate_up_delta[:, : slices * width]
        self.act = self.leg.act_pair
        self.down_rank_out = self.leg.down_rank_out
        self.b_gate_up = self.leg.b_gate_up
        self.a_down = self.leg.a_down
        self.aligned = self.leg.route(ROUTE_ALIGNED)
        self.fused = self.leg.route(ROUTE_FUSED_IDS)
        self.destination_offsets = (0, width) if slices == 2 else (0,)
        self.middle_specs = {
            fusion: FusedMiddleSpec(fusion=fusion, activation=self.activation)
            for fusion in ("b_act", "act_down_a", "full")
        }
        # FP32 admission oracle (host-syncing; never inside a timed thunk).
        # It reads only VALID bridge rows, so the poison stays unread.
        self.oracle_act, self.oracle_down = reference_fused_middle(
            bridge_gu=self.bridge,
            b_gate_up=self.b_gate_up,
            base_gu=self.base_gu,
            a_down=self.a_down,
            virtual_topk_ids=self.fused.virtual_topk_ids,
            activation=self.activation,
        )

    # ---- stage thunks ----

    def run_b(self, config, *, spec=None) -> None:
        run_lora_b(
            spec or _SPEC_B_ONE_LAUNCH,
            bridge=self.bridge,
            weight=self.b_gate_up,
            destination=self.gate_up_delta,
            routing=self.aligned,
            destination_offsets=self.destination_offsets,
            config=config,
        )

    def run_join(self, config=None) -> None:
        config = config if config is not None else JOIN_DEFAULT_CONFIG
        _materialized_join_kernel[(self.leg.num_pairs,)](
            self.gate_up_delta,
            self.base_gu,
            self.act,
            self.width,
            self.gate_up_delta.stride(0),
            self.base_gu.stride(0),
            self.act.stride(0),
            NUM_SLICES=self.num_slices,
            ACT_RELU2=self.activation == "relu2",
            BLOCK_W=int(config["BLOCK_W"]),
            num_warps=int(config["num_warps"]),
        )

    def run_down(self, config) -> None:
        run_lora_a(
            _SPEC_DOWN_GROUPED,
            input=self.act,
            weight=self.a_down,
            output=self.down_rank_out,
            routing=self.aligned,
            config=config,
        )

    def run_middle(self, fusion: str, config, routing=None) -> None:
        route = self.aligned if routing is None else routing
        spec = self.middle_specs[fusion]
        if fusion == "b_act":
            run_fused_middle(
                spec,
                base_gu=self.base_gu,
                act=self.act,
                routing=route,
                config=config,
                bridge_gu=self.bridge,
                b_gate_up=self.b_gate_up,
            )
        elif fusion == "act_down_a":
            run_fused_middle(
                spec,
                base_gu=self.base_gu,
                act=self.act,
                routing=route,
                config=config,
                gate_up_delta=self.gate_up_delta,
                a_down=self.a_down,
                down_rank_out=self.down_rank_out,
            )
        else:
            run_fused_middle(
                spec,
                base_gu=self.base_gu,
                act=self.act,
                routing=route,
                config=config,
                bridge_gu=self.bridge,
                b_gate_up=self.b_gate_up,
                a_down=self.a_down,
                down_rank_out=self.down_rank_out,
            )

    def run_cutedsl(self, plan) -> None:
        invoke_cutedsl_fused_middle(
            bridge_gu=self.bridge,
            b_gate_up=self.b_gate_up,
            base_gu=self.base_gu,
            a_down=self.a_down,
            routing=self.fused,
            act=self.act,
            down_rank_out=self.down_rank_out,
            plan=plan,
        )

    # ---- decided-arm composition ----

    def run_arm(self, arm: str, tuned: dict, plan=None) -> None:
        if arm == "m":
            self.run_b(tuned["m_b"])
            self.run_join(tuned["m_join"])
            self.run_down(tuned["m_down"])
        elif arm == "b_act":
            self.run_middle("b_act", tuned["b_act"])
            self.run_down(tuned["m_down"])
        elif arm == "act_down_a":
            self.run_b(tuned["m_b"])
            self.run_middle("act_down_a", tuned["act_down_a"])
        elif arm == "full":
            self.run_middle("full", tuned["full"])
        elif arm == "full_charged":
            self.run_middle(
                "full", tuned["full"], routing=self.leg.route(ROUTE_ALIGNED)
            )
        elif arm == "cutedsl":
            self.run_cutedsl(plan)
        elif arm == "cutedsl_charged":
            plan.build_metadata(verify=False)
            self.run_cutedsl(plan)
        else:
            raise ValueError(f"unknown arm {arm!r}")


def _require_close(observed, reference, *, has_signal: bool, label: str) -> None:
    """Signal gate, degrading to an exact-zero check on zero-valid routes
    (an all-zero reference cannot support a signal-relative gate)."""
    if has_signal:
        require_delta_close(
            observed.float(), reference.float(), gate_dtype=torch.bfloat16, label=label
        )
    elif not bool((observed == 0).all()):
        raise AssertionError(
            f"expected all-zero output on a zero-valid route [{label}]"
        )


def _check_middle_outputs(fixture: _MiddleFixture, *, label: str, down: str) -> None:
    """FP32-oracle admission of the common output boundary.

    ``down``: ``"none"`` (the arm under test writes no rank-out),
    ``"valid"`` (grouped down-A preserves sentinel rows), or
    ``"universal"`` (sentinel rows must be EXACT ZERO — contract 3).
    """
    require_finite(fixture.act, label=f"act finite [{label}]")
    require_delta_close(
        fixture.act.float(),
        fixture.oracle_act,
        gate_dtype=torch.bfloat16,
        label=f"act vs oracle [{label}]",
    )
    if down == "none":
        return
    if down == "universal":
        require_finite(fixture.down_rank_out, label=f"rank-out finite [{label}]")
        if fixture.has_valid:
            require_delta_close(
                fixture.down_rank_out.float(),
                fixture.oracle_down,
                gate_dtype=torch.bfloat16,
                label=f"rank-out vs oracle [{label}]",
            )
        if not bool((fixture.down_rank_out[~fixture.valid] == 0).all()):
            raise AssertionError(f"sentinel rank-out rows not exact zero [{label}]")
    elif down == "valid":
        if fixture.has_valid:
            require_delta_close(
                fixture.down_rank_out[fixture.valid].float(),
                fixture.oracle_down[fixture.valid],
                gate_dtype=torch.bfloat16,
                label=f"rank-out vs oracle (valid rows) [{label}]",
            )
    else:
        raise ValueError(f"unknown down mode {down!r}")


def _admit_arm(
    fixture: _MiddleFixture, arm: str, tuned: dict, plan, label: str
) -> None:
    """One un-timed poisoned-destination run vs the FP32 oracle."""
    fixture.act.fill_(float("nan"))
    fixture.down_rank_out.fill_(float("nan"))
    fixture.run_arm(arm, tuned, plan=plan)
    torch.cuda.synchronize()
    _check_middle_outputs(
        fixture,
        label=f"{arm} {label}",
        down="universal" if arm in _UNIVERSAL_DOWN_ARMS else "valid",
    )


def _build_cute_plan(fixture: _MiddleFixture, token_width: int):
    plan = build_cutedsl_fused_middle_plan(
        fused_route=fixture.fused,
        b_gate_up=fixture.b_gate_up,
        a_down=fixture.a_down,
        config=CutedslAConfig(token_width=token_width),
        activation=fixture.activation,
    )
    plan.build_metadata(verify=True)
    return plan


def _build_middle_case(
    device,
    *,
    geometry: str,
    preset: str,
    num_tokens: int,
    rank: int,
    seed: int,
    source_revision: str,
):
    ep_size, domain = VALIDITY_PRESETS[preset]
    return build_case(
        device=str(device),
        model_preset=GEOMETRY_PRESETS[geometry],
        topology=Topology(tp_size=8, ep_size=ep_size),
        adapter_cell=AdapterCell(
            active_adapters=4, include_base_rows=True, slot_capacity=8
        ),
        route_generator="iid",
        expert_id_domain=domain,
        num_tokens=num_tokens,
        active_rank=rank,
        seed=seed,
        source_revision=source_revision,
    )


def _best_config(*, grid, admit, thunk, timed, family, label, skips) -> dict:
    """Sweep one family: fail-closed skips (bench_common), numeric
    admission failures abort, fastest admissible config wins."""
    best_cfg, best_med = None, float("inf")
    skipped = 0
    for config in grid:
        try:
            admit(config)
        except Exception as error:
            reason = skip_reason(error)
            if reason is None:
                raise
            skipped += 1
            skips.append(
                {
                    "family": family,
                    "cell": label,
                    "config": _cfg_key(config),
                    "reason": reason,
                }
            )
            continue
        median = timed(lambda c=config: thunk(c), family, _cfg_key(config))
        if median < best_med:
            best_cfg, best_med = dict(config), median
    if best_cfg is None:
        raise RuntimeError(f"no admissible {family} config {label}")
    print(
        f"SWEEP {label} [{family}]: {_cfg_key(best_cfg)} "
        f"({best_med * 1e6:.1f}us, {skipped} skipped)",
        flush=True,
    )
    return best_cfg


def _sweep_cutedsl(
    *, fixture, case, label, suite, skips, cute_widths, timed
) -> dict | None:
    """Pick the fastest admissible token width; record per-GEMM ideal
    bounds.

    S5/6 verification, MAJOR: build failures were blanket-caught and
    demoted to sidecar skips, so the MANDATORY CuTeDSL arm could silently
    vanish from every cell while the run exited 0 — deferral by silent
    failure. Capability is pre-gated ONCE at startup; past that gate the
    plan builders' validators only fire on bench bugs, so any build
    failure now aborts the run (bench_common F3: numeric/contract
    failures are never skippable)."""
    best_width, best_med = None, float("inf")
    for width in cute_widths:
        plan = _build_cute_plan(fixture, width)
        fixture.act.fill_(float("nan"))
        fixture.down_rank_out.fill_(float("nan"))
        fixture.run_cutedsl(plan)
        torch.cuda.synchronize()
        _check_middle_outputs(
            fixture, label=f"cutedsl tw{width} {label}", down="universal"
        )
        median = timed(lambda p=plan: fixture.run_cutedsl(p), "cutedsl", f"tw{width}")
        for stage in ("slices", "down"):
            measure(
                lambda p=plan, s=stage: p.gemm_only(s),
                suite=suite,
                candidate=f"sweep_cutedsl_gemmbound_{stage}",
                boundary=BOUNDARY_ISOLATED,
                params={
                    "abi": "synthetic_pair_major",
                    "case_id": case.case_id,
                    "phase": "sweep",
                    "cell": label,
                    "family": "cutedsl_gemmbound",
                    "stage": stage,
                    "config": f"tw{width}",
                },
                graph_replay=True,
                warmup_iters=10,
                replay_iters=100,
            )
        if median < best_med:
            best_width, best_med = width, median
    if best_width is None:
        print(f"SWEEP {label} [cutedsl]: no admissible token width", flush=True)
        return None
    print(
        f"SWEEP {label} [cutedsl]: tw{best_width} ({best_med * 1e6:.1f}us)", flush=True
    )
    return {"token_width": best_width}


def _sweep_cell(
    *, fixture, case, label, suite, skips, cute_widths, sweep_params
) -> dict:
    """Tune every family on THIS cell's geometry; returns family -> config."""
    rank, width = fixture.rank, fixture.width

    def timed(fn, family, config_key):
        return measure(
            fn,
            suite=suite,
            candidate=f"sweep_{family}",
            boundary=BOUNDARY_PREPARED_INPUT,
            params={**sweep_params, "family": family, "config": config_key},
            graph_replay=True,
            warmup_iters=10,
            replay_iters=100,
        ).median_s

    tuned: dict = {}
    # Arm-M B stage: one-launch admitted against the S1-tested stock path
    # (which also certifies the zero-fill the join and AD rely on).
    fixture.gate_up_delta.fill_(71.0)
    fixture.run_b(PROVISIONAL_LAUNCH_CONFIG.lora_b, spec=_SPEC_B_STOCK)
    torch.cuda.synchronize()
    reference_delta = fixture.gate_up_delta.clone()
    if not bool((reference_delta[~fixture.valid] == 0).all()):
        raise AssertionError(f"stock B zero-fill broken {label}")

    def admit_b(config):
        fixture.gate_up_delta.fill_(-3.0)
        fixture.run_b(config)
        torch.cuda.synchronize()
        _require_close(
            fixture.gate_up_delta,
            reference_delta,
            has_signal=fixture.has_valid,
            label=f"m_b {label}",
        )

    tuned["m_b"] = _best_config(
        grid=_b_grid(rank),
        admit=admit_b,
        thunk=fixture.run_b,
        timed=timed,
        family="m_b",
        label=label,
        skips=skips,
    )

    # Correct delta so the join sweep reads real inputs; every join
    # config is oracle-admitted before timing (14th review: the join was
    # a fixed config, a weak control for the fusion comparison).
    fixture.run_b(tuned["m_b"])

    def admit_join(config):
        fixture.act.fill_(float("nan"))
        fixture.run_join(config)
        torch.cuda.synchronize()
        _check_middle_outputs(fixture, label=f"m join {label}", down="none")

    tuned["m_join"] = _best_config(
        grid=JOIN_GRID,
        admit=admit_join,
        thunk=lambda c: fixture.run_join(c),
        timed=timed,
        family="m_join",
        label=label,
        skips=skips,
    )
    fixture.act.fill_(float("nan"))
    fixture.run_join(tuned["m_join"])
    torch.cuda.synchronize()
    _check_middle_outputs(fixture, label=f"m join {label}", down="none")

    def admit_down(config):
        fixture.down_rank_out.fill_(float("nan"))
        fixture.run_down(config)
        torch.cuda.synchronize()
        _check_middle_outputs(fixture, label=f"m_down {label}", down="valid")

    tuned["m_down"] = _best_config(
        grid=_down_grid(rank, width),
        admit=admit_down,
        thunk=fixture.run_down,
        timed=timed,
        family="m_down",
        label=label,
        skips=skips,
    )

    def admit_middle(fusion, down_mode):
        def admit(config):
            fixture.act.fill_(float("nan"))
            if down_mode != "none":
                fixture.down_rank_out.fill_(float("nan"))
            fixture.run_middle(fusion, config)
            torch.cuda.synchronize()
            _check_middle_outputs(fixture, label=f"{fusion} {label}", down=down_mode)

        return admit

    for fusion, rank_loop, swizzle, down_mode in (
        ("b_act", True, True, "none"),
        ("act_down_a", False, False, "universal"),  # delta populated above
        ("full", True, False, "universal"),
    ):
        tuned[fusion] = _best_config(
            grid=_middle_grid(rank, rank_loop=rank_loop, swizzle=swizzle),
            admit=admit_middle(fusion, down_mode),
            thunk=lambda c, f=fusion: fixture.run_middle(f, c),
            timed=timed,
            family=fusion,
            label=label,
            skips=skips,
        )

    cute = _sweep_cutedsl(
        fixture=fixture,
        case=case,
        label=label,
        suite=suite,
        skips=skips,
        cute_widths=cute_widths,
        timed=timed,
    )
    if cute is not None:
        tuned["cutedsl"] = cute
    return tuned


def _candidate_name(arm: str, activation: str) -> str:
    base = arm.removesuffix("_charged")
    if base == "m":
        name = f"middle_materialized_{activation}"
    elif base == "cutedsl":
        name = f"middle_cutedsl_{activation}"
    else:
        name = f"middle_{base}_{activation}"  # == FusedMiddleSpec.key()
    return name + ("_charged" if arm.endswith("_charged") else "")


def _arm_config_key(arm: str, tuned: dict, cute_width: int | None) -> str:
    if arm == "m":
        return (
            f"b:{_cfg_key(tuned['m_b'])}|join:{_cfg_key(tuned['m_join'])}"
            f"|down:{_cfg_key(tuned['m_down'])}"
        )
    if arm == "b_act":
        return f"ba:{_cfg_key(tuned['b_act'])}|down:{_cfg_key(tuned['m_down'])}"
    if arm == "act_down_a":
        return f"b:{_cfg_key(tuned['m_b'])}|ad:{_cfg_key(tuned['act_down_a'])}"
    if arm in ("full", "full_charged"):
        return _cfg_key(tuned["full"])
    return f"tw{cute_width}"


def _decided_params(
    *,
    case,
    fixture,
    arm,
    tuned,
    cute_width,
    plan,
    seed,
    repeat,
    regime,
    geometry,
    preset,
    num_tokens,
) -> dict:
    """Evidence extras per §65.1: eliminated bytes and launch counts.
    Staged CUTE buffers are recorded raw, never netted against the
    eliminated pair-domain delta."""
    pairs = fixture.leg.num_pairs
    delta_roundtrip = 2 * pairs * fixture.num_slices * fixture.width * 2
    act_reread = pairs * fixture.width * 2
    eliminated = {
        "m": 0,
        "b_act": delta_roundtrip,
        "act_down_a": act_reread,
        "full": delta_roundtrip + act_reread,
        "full_charged": delta_roundtrip + act_reread,
        "cutedsl": delta_roundtrip,
        "cutedsl_charged": delta_roundtrip,
    }[arm]
    params = {
        "abi": "synthetic_pair_major",
        "case_id": case.case_id,
        "phase": "decided",
        "geometry": geometry,
        "model_preset": case.model_preset,
        "activation": fixture.activation,
        "validity": preset,
        "T": num_tokens,
        "P": pairs,
        "rank": fixture.rank,
        "width": fixture.width,
        "seed": seed,
        "repeat": repeat,
        "regime": regime,
        "arm": arm,
        "arm_config": _arm_config_key(arm, tuned, cute_width),
        "launches": _LAUNCHES[arm],
        "plus_route_build_in_thunk": arm.endswith("_charged"),
        "bytes_delta_roundtrip": delta_roundtrip,
        "bytes_act_reread": act_reread,
        "bytes_eliminated": eliminated,
    }
    if plan is not None and arm.startswith("cutedsl"):
        params["cutedsl"] = {
            "token_width": cute_width,
            "m_max": plan.m_max,
            "num_groups": plan.num_groups,
            "staged_buffer_bytes": 2
            * (
                plan.staged_bridge.numel()
                + plan.c_slices.numel()
                + plan.staged_act.numel()
                + plan.c_down.numel()
            ),
        }
    return params


def _decided_cell(
    *,
    device,
    geometry,
    preset,
    rank,
    num_tokens,
    suite,
    skips,
    best,
    samples,
) -> None:
    regime = regime_of(num_tokens)
    tuned = load_decided_tuned(best, (geometry, preset, rank, regime))
    cute_key = (geometry, preset, rank, regime, "cutedsl")
    cute_width = best[cute_key]["token_width"] if cute_key in best else None
    modes = (True,) if num_tokens <= DECODE_T_MAX else (True, False)
    for seed in SEEDS:
        case = _build_middle_case(
            device,
            geometry=geometry,
            preset=preset,
            num_tokens=num_tokens,
            rank=rank,
            seed=seed,
            source_revision=suite.source_revision,
        )
        fixture = _MiddleFixture(case, device)
        label = f"{geometry} {preset} r{rank} T={num_tokens} s{seed}"
        # S5/6 verification, MAJOR: a build failure here used to drop the
        # MANDATORY cute arm from the decided cell via a sidecar entry.
        # Capability is pre-gated at startup; past it, failures abort.
        plan = None
        if cute_width is not None:
            plan = _build_cute_plan(fixture, cute_width)
        arms = [arm for arm in ARMS if arm != "cutedsl" or plan is not None]
        arms += [
            arm
            for arm in CHARGED_ARMS
            if not arm.startswith("cutedsl") or plan is not None
        ]
        for arm in arms:
            _admit_arm(fixture, arm, tuned, plan, label)
        cell = (geometry, preset, rank, num_tokens)
        for graph in modes:
            mode = "graph" if graph else "eager"
            for repeat in range(REPEATS):
                names = arms if repeat % 2 == 0 else arms[::-1]
                for arm in names:
                    record = measure(
                        lambda a=arm: fixture.run_arm(a, tuned, plan=plan),
                        suite=suite,
                        candidate=_candidate_name(arm, fixture.activation),
                        boundary=BOUNDARIES[arm],
                        params=_decided_params(
                            case=case,
                            fixture=fixture,
                            arm=arm,
                            tuned=tuned,
                            cute_width=cute_width,
                            plan=plan,
                            seed=seed,
                            repeat=repeat,
                            regime=regime,
                            geometry=geometry,
                            preset=preset,
                            num_tokens=num_tokens,
                        ),
                        graph_replay=graph,
                    )
                    samples[(*cell, mode)][arm].append(record.median_s)


def _parse_cli() -> argparse.Namespace:
    """CLI parsing + fail-fast preflight against every arm's bounds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,64,128")
    parser.add_argument("--validity", default="dense,ep8")
    parser.add_argument("--geometries", default="gated,relu2")
    parser.add_argument(
        "--cutedsl-token-widths",
        default="auto",
        help="comma list of masked-GEMM token widths, 'auto' (all widths "
        "this arch supports), or 'off' (drop the CUTE arm)",
    )
    arguments = parser.parse_args()
    arguments.rank_list = tuple(int(rank) for rank in arguments.ranks.split(","))
    for rank in arguments.rank_list:
        if not 16 <= rank <= 128:
            raise ValueError(
                f"rank {rank} outside [16, 128] (tl.dot minimum / the AD-FULL "
                "register-resident bound MAX_DOWN_RANK)"
            )
        if rank % 8:
            raise ValueError(f"rank {rank} must be a multiple of 8 (CuTeDSL TMA)")
    arguments.preset_list = tuple(arguments.validity.split(","))
    for preset in arguments.preset_list:
        if preset not in VALIDITY_PRESETS:
            raise ValueError(f"unknown validity preset {preset!r}")
    arguments.geometry_list = tuple(arguments.geometries.split(","))
    for geometry in arguments.geometry_list:
        if geometry not in GEOMETRY_PRESETS:
            raise ValueError(f"unknown geometry {geometry!r}")
    return arguments


def main() -> int:
    arguments = _parse_cli()
    device = torch.device(arguments.device)
    require_writable_destination(arguments.output)
    # S5/6 verification (m8/m9): supported_token_widths() returns a
    # non-empty SM90 width list on ANY pre-sm100 device including sm8x,
    # so 'auto' would enable CuTeDSL where the masked GEMM cannot exist.
    # Gate capability HERE, once, exactly like bench_finalize.
    torch.cuda.set_device(device)
    major, _ = torch.cuda.get_device_capability(device)
    if arguments.cutedsl_token_widths != "off" and major < 9:
        raise ValueError(
            f"CuTeDSL arms are mandatory but this device is sm{major}x < "
            "sm90; run on a supported device or pass "
            "--cutedsl-token-widths off for a Triton-only diagnostic run"
        )
    ranks = arguments.rank_list
    presets = arguments.preset_list
    geometries = arguments.geometry_list
    if arguments.cutedsl_token_widths == "off":
        cute_widths: tuple[int, ...] = ()
    elif arguments.cutedsl_token_widths == "auto":
        cute_widths = supported_token_widths(device)
    else:
        cute_widths = tuple(
            int(width) for width in arguments.cutedsl_token_widths.split(",")
        )
    # 14th S5/6 review: the ABI gap is LABELED into the artifact identity
    # so these suites can never be mistaken for qualification evidence.
    suite = new_suite(
        "fused_middle_synthetic_pair_abi_v1",
        source_revision=arguments.source_revision,
    )
    skips: list[dict] = []

    def cell_presets(geometry: str) -> tuple[str, ...]:
        return tuple(p for p in presets if p in GEOMETRY_VALIDITY[geometry])

    # ---- SWEEP per (geometry, validity, rank, regime).
    best: dict = {}
    for geometry in geometries:
        for preset in cell_presets(geometry):
            for rank in ranks:
                for regime, num_tokens in SWEEP_T.items():
                    case = _build_middle_case(
                        device,
                        geometry=geometry,
                        preset=preset,
                        num_tokens=num_tokens,
                        rank=rank,
                        seed=SEEDS[0],
                        source_revision=suite.source_revision,
                    )
                    fixture = _MiddleFixture(case, device)
                    label = f"{geometry} {preset} r{rank} {regime}(T={num_tokens})"
                    tuned = _sweep_cell(
                        fixture=fixture,
                        case=case,
                        label=label,
                        suite=suite,
                        skips=skips,
                        cute_widths=cute_widths,
                        sweep_params={
                            "abi": "synthetic_pair_major",
                            "case_id": case.case_id,
                            "phase": "sweep",
                            "geometry": geometry,
                            "validity": preset,
                            "T": num_tokens,
                            "rank": rank,
                            "regime": regime,
                        },
                    )
                    for family, config in tuned.items():
                        best[(geometry, preset, rank, regime, family)] = config

    # ---- DECIDED: seeded interleaved comparison, tuned on this geometry.
    samples: dict = defaultdict(lambda: defaultdict(list))
    for geometry in geometries:
        for preset in cell_presets(geometry):
            for rank in ranks:
                for num_tokens in T_GRID:
                    _decided_cell(
                        device=device,
                        geometry=geometry,
                        preset=preset,
                        rank=rank,
                        num_tokens=num_tokens,
                        suite=suite,
                        skips=skips,
                        best=best,
                        samples=samples,
                    )

    for cell in sorted(samples):
        for arm_a, arm_b in DECIDED_PAIRS:
            cell_samples = samples[cell]
            if arm_a not in cell_samples or arm_b not in cell_samples:
                continue
            if len(cell_samples[arm_a]) != len(cell_samples[arm_b]):
                print(f"UNPAIRED {cell} {arm_a}/{arm_b}; no decision")
                continue
            decision = decide_cell(
                arm_a=arm_a,
                samples_a=cell_samples[arm_a],
                arm_b=arm_b,
                samples_b=cell_samples[arm_b],
            )
            print(
                f"{cell[4]:5s} {cell[0]:5s} {cell[1]:5s} r{cell[2]:<4d} "
                f"T={cell[3]:<5d} {arm_a}/{arm_b:18s} "
                f"geo(a/b)={decision.geo_a_over_b:.3f} -> "
                f"{decision.winner or 'tied'}"
            )

    write_skip_sidecar(arguments.output, skips)
    # S5/6 verification, MAJOR: the CuTeDSL arms are MANDATORY. If the
    # startup capability gate enabled them, a publication without a single
    # cute record means the arm silently vanished — refuse to publish
    # Triton-only evidence under a complete-looking exit.
    if cute_widths and not any(
        "cutedsl" in record.candidate for record in suite.records
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
