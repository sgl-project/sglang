"""Step-5 fused-middle candidates (benchmark tier, plan §65.1).

The MIDDLE of one MoE LoRA leg is gate/up-B -> activation join -> down-A.
The materialized baseline (arm M in the bench) runs three launches and
round-trips ``gate_up_delta`` [P, 2W] and ``act`` [P, W] through HBM. The
ladder here removes those boundaries one at a time so gate 5 can attribute
the win:

* **b_act** (arm BA): fused gate/up-B + activation. Per (m-block, W-tile)
  over the aligned plan: two rank dots (gate slice, up slice) + the base
  tiles + the activation, one ``act`` store. Kills the delta buffer.
* **act_down_a** (arm AD): fused activation + down-A. Reads a
  MATERIALIZED ``gate_up_delta`` + the base; per m-block a SERIAL W-tile
  loop computes each act tile, stores it, and accumulates
  ``down_rank_out`` [BLOCK_M, R2] in FP32 registers — fixed tile order,
  deterministic by construction. Kills the act re-read.
* **full** (arm FULL): B + activation + down-A in ONE kernel — the same
  serial W-tile loop, but each tile's delta comes from in-register rank
  dots over ``bridge_gu`` x ``b_gate_up``. No delta buffer exists at all.
  Register bound: the FP32 accumulator is [BLOCK_M, R2]; ``MAX_DOWN_RANK``
  fail-closes the arm at R2 <= 128 instead of silently spilling.

THE CONTRACT every arm satisfies (what makes buffer reuse under one CUDA
graph safe, pinned by the registered tests):

* ``act`` [P, W] is written for EVERY pair row on every call — it is also
  the base W2 input, so it must exist at the common output boundary.
  Sentinel rows (``virtual_expert_id == -1``: base tokens, invalid pairs,
  non-owned experts) get ``activation(base)`` with delta = 0, and their
  bridge/delta rows are NEVER read (they may hold poison).
* ``down_rank_out`` [P, R2] (AD/FULL only) is written for every pair row:
  valid pairs get ``act_tile @ a_down[veid]^T`` accumulated in FP32 over
  the serial W-tile loop; sentinel rows get EXACT ZERO. The down dot
  consumes the STORED act tiles (cast to the act dtype first), so a fused
  arm reproduces the materialized baseline's numbers, not better ones.

Activations (``NUM_SLICES``/``ACT_RELU2`` constexpr): ``silu_mul`` is the
gated two-slice form — ``act = silu(base_g + delta_g) * (base_u +
delta_u)`` with slice s of ``bridge_gu`` at columns [s*R, (s+1)*R), of
``b_gate_up`` at rows [s*W, (s+1)*W), of ``base_gu`` at columns
[s*W, (s+1)*W); ``relu2`` is the non-gated single-slice guardrail —
``act = relu(base + delta)^2`` with W the total width.

Weight layouts follow the leg fixture (bench_lora_a): ``b_gate_up`` is
[G, NUM_SLICES*W, R] and ``a_down`` is [G, R2, W] with G = max_loras *
lora_experts_per_adapter, exactly the flattened case tensors the A/B
candidates consume.

Config keys: ``BLOCK_SIZE_W`` (the W tile), ``BLOCK_SIZE_K`` (the rank
loop of the B dots; b_act/full only), ``GROUP_SIZE_M`` (the b_act grid
swizzle; the serial arms launch one program per m-block and ignore it),
``num_warps``, ``num_stages``. BLOCK_SIZE_M is the routing block size.
"""

from __future__ import annotations

from collections.abc import Mapping

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.lora.sgl_lora.routing import RouteView

FUSIONS = ("b_act", "act_down_a", "full")
MIDDLE_ACTIVATIONS = ("silu_mul", "relu2")
# The AD/FULL accumulator is [BLOCK_M, R2] FP32 in registers; past rank
# 128 it spills and the arm's premise (register-resident down-A) is a
# lie. Fail closed rather than benchmark a spill.
MAX_DOWN_RANK = 128
# The register-resident premise in ACCUMULATOR VALUES: the known-good
# baseline is block_size 16 x R2_padded 128 = 2048 fp32 accum values per
# program (S5/6 verification, m2 — R2 alone did not bound the accumulator).
MAX_ACCUM_VALUES = 16 * 128

FUSED_B_ACT_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_W": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}
# The serial arms (act_down_a, full) carry the FP32 accumulator across
# the W-tile loop; two stages keeps the pipeliner off the loop-carried
# dependency. GROUP_SIZE_M is present for config-grid uniformity only.
FUSED_MIDDLE_SERIAL_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_W": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}


class FusedMiddleSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One point in the fused-middle ladder."""

    fusion: str
    activation: str

    def __post_init__(self):
        for field_name, (value, vocabulary) in {
            "fusion": (self.fusion, FUSIONS),
            "activation": (self.activation, MIDDLE_ACTIVATIONS),
        }.items():
            if value not in vocabulary:
                raise ValueError(f"{field_name}={value!r} is not one of {vocabulary}")

    def key(self) -> str:
        return f"middle_{self.fusion}_{self.activation}"


def _validate_middle_call(
    *,
    activation: str,
    base_gu: torch.Tensor,
    act: torch.Tensor,
    routing: RouteView,
    bridge_gu: torch.Tensor | None = None,
    b_gate_up: torch.Tensor | None = None,
    gate_up_delta: torch.Tensor | None = None,
    a_down: torch.Tensor | None = None,
    down_rank_out: torch.Tensor | None = None,
) -> tuple[int, int, int, int]:
    """Fail-closed contract check; returns (num_slices, width, rank, down_rank)."""
    if activation not in MIDDLE_ACTIVATIONS:
        raise ValueError(
            f"activation={activation!r} is not one of {MIDDLE_ACTIVATIONS}"
        )
    if (bridge_gu is None) != (b_gate_up is None):
        raise ValueError("bridge_gu and b_gate_up travel together")
    if (a_down is None) != (down_rank_out is None):
        raise ValueError("a_down and down_rank_out travel together")
    num_slices = 2 if activation == "silu_mul" else 1
    num_pairs = routing.topk_ids.numel()
    if act.ndim != 2 or act.shape[0] != num_pairs or act.shape[1] < 1:
        raise ValueError(f"act must be [{num_pairs}, W>=1], got {tuple(act.shape)}")
    width = act.shape[1]
    if base_gu.shape != (num_pairs, num_slices * width):
        raise ValueError(
            f"base_gu must be {(num_pairs, num_slices * width)}, got "
            f"{tuple(base_gu.shape)}"
        )
    expected_groups = routing.max_loras * routing.lora_experts_per_adapter
    rank = 0
    if b_gate_up is not None:
        if b_gate_up.ndim != 3 or b_gate_up.shape[:2] != (
            expected_groups,
            num_slices * width,
        ):
            raise ValueError(
                f"b_gate_up must be [{expected_groups}, {num_slices * width}, R], "
                f"got {tuple(b_gate_up.shape)}"
            )
        rank = b_gate_up.shape[2]
        if rank < 1:
            raise ValueError("b_gate_up rank must be >= 1")
        if bridge_gu.shape != (num_pairs, num_slices * rank):
            raise ValueError(
                f"bridge_gu must be {(num_pairs, num_slices * rank)}, got "
                f"{tuple(bridge_gu.shape)}"
            )
        if bridge_gu.dtype != b_gate_up.dtype:
            raise ValueError(
                f"the delta dot needs one dtype: bridge_gu {bridge_gu.dtype} "
                f"vs b_gate_up {b_gate_up.dtype}"
            )
    if gate_up_delta is not None and gate_up_delta.shape != (
        num_pairs,
        num_slices * width,
    ):
        raise ValueError(
            f"gate_up_delta must be {(num_pairs, num_slices * width)}, got "
            f"{tuple(gate_up_delta.shape)}"
        )
    down_rank = 0
    if a_down is not None:
        if (
            a_down.ndim != 3
            or a_down.shape[0] != expected_groups
            or (a_down.shape[2] != width)
        ):
            raise ValueError(
                f"a_down must be [{expected_groups}, R2, {width}], got "
                f"{tuple(a_down.shape)}"
            )
        down_rank = a_down.shape[1]
        acc_values = routing.block_size * triton.next_power_of_2(max(down_rank, 1))
        if acc_values > MAX_ACCUM_VALUES:
            raise ValueError(
                f"acc[BLOCK_M={routing.block_size}, R2_padded="
                f"{triton.next_power_of_2(max(down_rank, 1))}] = {acc_values} "
                f"fp32 values exceeds the register-resident premise cap "
                f"{MAX_ACCUM_VALUES}; this arm's regime ends where spills "
                "begin (S5/6 verification: the old R2<=128 check admitted "
                "spilled shapes at larger routing block sizes)"
            )
        if not 1 <= down_rank <= MAX_DOWN_RANK:
            raise ValueError(
                f"down rank {down_rank} exceeds the register-resident bound "
                f"{MAX_DOWN_RANK} (acc[BLOCK_M, R2] FP32); this arm's regime "
                "ends where spills begin"
            )
        if down_rank_out.shape != (num_pairs, down_rank):
            raise ValueError(
                f"down_rank_out must be {(num_pairs, down_rank)}, got "
                f"{tuple(down_rank_out.shape)}"
            )
        if act.dtype != a_down.dtype:
            raise ValueError(
                f"the down dot consumes STORED act tiles: act {act.dtype} "
                f"vs a_down {a_down.dtype} must be one dtype"
            )
    present = (base_gu, act, bridge_gu, b_gate_up, gate_up_delta, a_down, down_rank_out)
    devices = {t.device for t in present if t is not None} | {routing.topk_ids.device}
    if len(devices) != 1:
        raise ValueError(f"tensors span devices {sorted(map(str, devices))}")
    return num_slices, width, rank, down_rank


def _require_positive_group(group_size_m: int) -> int:
    """S5/6 verification: GROUP_SIZE_M=0 reached a device-side division
    inside the swizzle instead of a contract error."""
    if group_size_m < 1:
        raise ValueError(f"GROUP_SIZE_M must be >= 1, got {group_size_m}")
    return group_size_m


def _require_dot_geometry(
    *, block_size_m: int, block_size_w: int, block_size_k: int | None = None
) -> None:
    """tl.dot rejects tiles under 16; surface that as a config error."""
    checks = {"routing block_size": block_size_m, "BLOCK_SIZE_W": block_size_w}
    if block_size_k is not None:
        checks["BLOCK_SIZE_K"] = block_size_k
    for name, value in checks.items():
        if value < 16:
            raise ValueError(f"{name}={value} is below the tl.dot minimum of 16")


@triton.jit
def _activation_tile(gate, up, ACT_RELU2: tl.constexpr):
    """FP32 activation join: SwiGLU (gated) or ReLU^2 (non-gated)."""
    if ACT_RELU2:
        clamped = tl.maximum(gate, 0.0)
        result = clamped * clamped
    else:
        result = gate * tl.sigmoid(gate) * up
    return result


@triton.jit
def _delta_slice_dot(
    bridge_ptr,
    weight_group_ptr,
    pair_ids,
    pair_mask,
    w_offsets,
    w_mask,
    stride_rm,
    stride_rk,
    stride_bn,
    stride_bk,
    slice_id: tl.constexpr,
    RANK: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One slice's (bridge x B[veid]^T) delta tile, FP32 accumulated."""
    delta = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_W), dtype=tl.float32)
    for k_begin in range(0, RANK, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < RANK
        lhs = tl.load(
            bridge_ptr
            + pair_ids[:, None] * stride_rm
            + (slice_id * RANK + k_offsets)[None, :] * stride_rk,
            mask=pair_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            weight_group_ptr
            + (slice_id * W + w_offsets)[None, :] * stride_bn
            + k_offsets[:, None] * stride_bk,
            mask=w_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        delta += tl.dot(lhs, rhs, out_dtype=tl.float32)
    return delta


@triton.jit
def _delta_pair_dot(
    bridge_ptr,
    group_ptr,
    pair_ids,
    pair_mask,
    w_offsets,
    w_mask,
    stride_rm,
    stride_rk,
    stride_bn,
    stride_bk,
    NUM_SLICES: tl.constexpr,
    RANK: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """(delta_gate, delta_up) tiles; delta_up is zeros when non-gated."""
    delta_gate = _delta_slice_dot(
        bridge_ptr,
        group_ptr,
        pair_ids,
        pair_mask,
        w_offsets,
        w_mask,
        stride_rm,
        stride_rk,
        stride_bn,
        stride_bk,
        slice_id=0,
        RANK=RANK,
        W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    if NUM_SLICES == 2:
        delta_up = _delta_slice_dot(
            bridge_ptr,
            group_ptr,
            pair_ids,
            pair_mask,
            w_offsets,
            w_mask,
            stride_rm,
            stride_rk,
            stride_bn,
            stride_bk,
            slice_id=1,
            RANK=RANK,
            W=W,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    else:
        delta_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_W), dtype=tl.float32)
    return delta_gate, delta_up


@triton.jit
def _fused_b_act_kernel(
    bridge_ptr,
    b_ptr,
    base_ptr,
    act_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    stride_rm,
    stride_rk,
    stride_bg,
    stride_bn,
    stride_bk,
    stride_pm,
    stride_pn,
    stride_am,
    stride_an,
    W: tl.constexpr,
    RANK: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ACT_RELU2: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Arm BA: per (m-block, W-tile) delta dots + base + activation -> act.

    Sentinel blocks skip the dots (their bridge rows may hold poison) and
    still store ``activation(base)`` — act is universal.
    """
    pid = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    num_pid_w: tl.constexpr = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    programs_per_group = GROUP_SIZE_M * num_pid_w
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_w = (pid % programs_per_group) // group_size_m
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    virtual_expert_id = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)

    w_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W).to(tl.int64)
    w_mask = w_offsets < W
    load_mask = pair_mask[:, None] & w_mask[None, :]

    base_gate = tl.load(
        base_ptr + pair_ids[:, None] * stride_pm + w_offsets[None, :] * stride_pn,
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)
    delta_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_W), dtype=tl.float32)
    delta_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_W), dtype=tl.float32)
    if virtual_expert_id != -1:
        dot_gate, dot_up = _delta_pair_dot(
            bridge_ptr,
            b_ptr + virtual_expert_id * stride_bg,
            pair_ids,
            pair_mask,
            w_offsets,
            w_mask,
            stride_rm,
            stride_rk,
            stride_bn,
            stride_bk,
            NUM_SLICES=NUM_SLICES,
            RANK=RANK,
            W=W,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        delta_gate += dot_gate
        delta_up += dot_up
    gate = base_gate + delta_gate
    if NUM_SLICES == 2:
        base_up = tl.load(
            base_ptr
            + pair_ids[:, None] * stride_pm
            + (W + w_offsets)[None, :] * stride_pn,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        up = base_up + delta_up
    else:
        up = gate
    act_tile = _activation_tile(gate, up, ACT_RELU2)
    tl.store(
        act_ptr + pair_ids[:, None] * stride_am + w_offsets[None, :] * stride_an,
        act_tile.to(act_ptr.dtype.element_ty),
        mask=load_mask,
    )


@triton.jit
def _act_down_a_kernel(
    delta_ptr,
    base_ptr,
    a_ptr,
    act_ptr,
    rank_out_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    stride_dm,
    stride_dn,
    stride_pm,
    stride_pn,
    stride_ag,
    stride_ar,
    stride_aw,
    stride_am,
    stride_an,
    stride_om,
    stride_on,
    W: tl.constexpr,
    RANK_DOWN: tl.constexpr,
    RANK_DOWN_PADDED: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ACT_RELU2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """Arm AD: one program per m-block; SERIAL W-tile loop, deterministic.

    Each tile: act = activation(base + materialized delta), stored, then
    ``acc += act_tile @ a_down[veid]^T`` in FP32. Sentinel blocks never
    read the delta buffer (it may hold poison), store activation(base),
    and leave acc at zero — the exact-zero rank-out contract.
    """
    pid_m = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return
    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    virtual_expert_id = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)

    r_offsets = tl.arange(0, RANK_DOWN_PADDED).to(tl.int64)
    r_mask = r_offsets < RANK_DOWN
    acc = tl.zeros((BLOCK_SIZE_M, RANK_DOWN_PADDED), dtype=tl.float32)
    for w_begin in range(0, W, BLOCK_SIZE_W):
        w_offsets = w_begin + tl.arange(0, BLOCK_SIZE_W).to(tl.int64)
        w_mask = w_offsets < W
        load_mask = pair_mask[:, None] & w_mask[None, :]
        base_gate = tl.load(
            base_ptr + pair_ids[:, None] * stride_pm + w_offsets[None, :] * stride_pn,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        delta_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_W), dtype=tl.float32)
        delta_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_W), dtype=tl.float32)
        if virtual_expert_id != -1:
            delta_gate += tl.load(
                delta_ptr
                + pair_ids[:, None] * stride_dm
                + w_offsets[None, :] * stride_dn,
                mask=load_mask,
                other=0.0,
            ).to(tl.float32)
            if NUM_SLICES == 2:
                delta_up += tl.load(
                    delta_ptr
                    + pair_ids[:, None] * stride_dm
                    + (W + w_offsets)[None, :] * stride_dn,
                    mask=load_mask,
                    other=0.0,
                ).to(tl.float32)
        gate = base_gate + delta_gate
        if NUM_SLICES == 2:
            base_up = tl.load(
                base_ptr
                + pair_ids[:, None] * stride_pm
                + (W + w_offsets)[None, :] * stride_pn,
                mask=load_mask,
                other=0.0,
            ).to(tl.float32)
            up = base_up + delta_up
        else:
            up = gate
        act_tile = _activation_tile(gate, up, ACT_RELU2)
        act_stored = act_tile.to(act_ptr.dtype.element_ty)
        tl.store(
            act_ptr + pair_ids[:, None] * stride_am + w_offsets[None, :] * stride_an,
            act_stored,
            mask=load_mask,
        )
        if virtual_expert_id != -1:
            a_tile = tl.load(
                a_ptr
                + virtual_expert_id * stride_ag
                + r_offsets[None, :] * stride_ar
                + w_offsets[:, None] * stride_aw,
                mask=w_mask[:, None] & r_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(act_stored, a_tile, out_dtype=tl.float32)
    tl.store(
        rank_out_ptr + pair_ids[:, None] * stride_om + r_offsets[None, :] * stride_on,
        acc.to(rank_out_ptr.dtype.element_ty),
        mask=pair_mask[:, None] & r_mask[None, :],
    )


@triton.jit
def _fused_middle_kernel(
    bridge_ptr,
    b_ptr,
    base_ptr,
    a_ptr,
    act_ptr,
    rank_out_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    stride_rm,
    stride_rk,
    stride_bg,
    stride_bn,
    stride_bk,
    stride_pm,
    stride_pn,
    stride_ag,
    stride_ar,
    stride_aw,
    stride_am,
    stride_an,
    stride_om,
    stride_on,
    W: tl.constexpr,
    RANK: tl.constexpr,
    RANK_DOWN: tl.constexpr,
    RANK_DOWN_PADDED: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ACT_RELU2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Arm FULL: B + activation + down-A in one launch, serial W-tile loop.

    Identical to arm AD except each tile's delta is computed in registers
    from ``bridge_gu`` x ``b_gate_up`` — no delta buffer exists anywhere.
    """
    pid_m = tl.program_id(0)
    num_pairs_post_padded = tl.load(num_pairs_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_pairs_post_padded:
        return
    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    virtual_expert_id = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)
    group_ptr = b_ptr + tl.maximum(virtual_expert_id, 0) * stride_bg

    r_offsets = tl.arange(0, RANK_DOWN_PADDED).to(tl.int64)
    r_mask = r_offsets < RANK_DOWN
    acc = tl.zeros((BLOCK_SIZE_M, RANK_DOWN_PADDED), dtype=tl.float32)
    for w_begin in range(0, W, BLOCK_SIZE_W):
        w_offsets = w_begin + tl.arange(0, BLOCK_SIZE_W).to(tl.int64)
        w_mask = w_offsets < W
        load_mask = pair_mask[:, None] & w_mask[None, :]
        base_gate = tl.load(
            base_ptr + pair_ids[:, None] * stride_pm + w_offsets[None, :] * stride_pn,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        delta_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_W), dtype=tl.float32)
        delta_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_W), dtype=tl.float32)
        if virtual_expert_id != -1:
            dot_gate, dot_up = _delta_pair_dot(
                bridge_ptr,
                group_ptr,
                pair_ids,
                pair_mask,
                w_offsets,
                w_mask,
                stride_rm,
                stride_rk,
                stride_bn,
                stride_bk,
                NUM_SLICES=NUM_SLICES,
                RANK=RANK,
                W=W,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_W=BLOCK_SIZE_W,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
            )
            delta_gate += dot_gate
            delta_up += dot_up
        gate = base_gate + delta_gate
        if NUM_SLICES == 2:
            base_up = tl.load(
                base_ptr
                + pair_ids[:, None] * stride_pm
                + (W + w_offsets)[None, :] * stride_pn,
                mask=load_mask,
                other=0.0,
            ).to(tl.float32)
            up = base_up + delta_up
        else:
            up = gate
        act_tile = _activation_tile(gate, up, ACT_RELU2)
        act_stored = act_tile.to(act_ptr.dtype.element_ty)
        tl.store(
            act_ptr + pair_ids[:, None] * stride_am + w_offsets[None, :] * stride_an,
            act_stored,
            mask=load_mask,
        )
        if virtual_expert_id != -1:
            a_tile = tl.load(
                a_ptr
                + virtual_expert_id * stride_ag
                + r_offsets[None, :] * stride_ar
                + w_offsets[:, None] * stride_aw,
                mask=w_mask[:, None] & r_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(act_stored, a_tile, out_dtype=tl.float32)
    tl.store(
        rank_out_ptr + pair_ids[:, None] * stride_om + r_offsets[None, :] * stride_on,
        acc.to(rank_out_ptr.dtype.element_ty),
        mask=pair_mask[:, None] & r_mask[None, :],
    )


def invoke_fused_b_act(
    bridge_gu: torch.Tensor,
    b_gate_up: torch.Tensor,
    base_gu: torch.Tensor,
    act: torch.Tensor,
    routing: RouteView,
    *,
    activation: str,
    config: Mapping[str, int],
) -> None:
    num_slices, width, rank, _ = _validate_middle_call(
        activation=activation,
        base_gu=base_gu,
        act=act,
        routing=routing,
        bridge_gu=bridge_gu,
        b_gate_up=b_gate_up,
    )
    if routing.topk_ids.numel() == 0:
        return
    block_size_w = int(config["BLOCK_SIZE_W"])
    block_size_k = int(config["BLOCK_SIZE_K"])
    _require_dot_geometry(
        block_size_m=routing.block_size,
        block_size_w=block_size_w,
        block_size_k=block_size_k,
    )
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_w_tiles = triton.cdiv(width, block_size_w)
    _fused_b_act_kernel[(num_m_blocks * num_w_tiles,)](
        bridge_gu,
        b_gate_up,
        base_gu,
        act,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        routing.topk_ids.numel(),
        bridge_gu.stride(0),
        bridge_gu.stride(1),
        b_gate_up.stride(0),
        b_gate_up.stride(1),
        b_gate_up.stride(2),
        base_gu.stride(0),
        base_gu.stride(1),
        act.stride(0),
        act.stride(1),
        W=width,
        RANK=rank,
        NUM_SLICES=num_slices,
        ACT_RELU2=activation == "relu2",
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_W=block_size_w,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=_require_positive_group(int(config["GROUP_SIZE_M"])),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def invoke_act_down_a(
    gate_up_delta: torch.Tensor,
    base_gu: torch.Tensor,
    a_down: torch.Tensor,
    act: torch.Tensor,
    down_rank_out: torch.Tensor,
    routing: RouteView,
    *,
    activation: str,
    config: Mapping[str, int],
) -> None:
    num_slices, width, _, down_rank = _validate_middle_call(
        activation=activation,
        base_gu=base_gu,
        act=act,
        routing=routing,
        gate_up_delta=gate_up_delta,
        a_down=a_down,
        down_rank_out=down_rank_out,
    )
    if routing.topk_ids.numel() == 0:
        return
    block_size_w = int(config["BLOCK_SIZE_W"])
    _require_dot_geometry(block_size_m=routing.block_size, block_size_w=block_size_w)
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    _act_down_a_kernel[(num_m_blocks,)](
        gate_up_delta,
        base_gu,
        a_down,
        act,
        down_rank_out,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        routing.topk_ids.numel(),
        gate_up_delta.stride(0),
        gate_up_delta.stride(1),
        base_gu.stride(0),
        base_gu.stride(1),
        a_down.stride(0),
        a_down.stride(1),
        a_down.stride(2),
        act.stride(0),
        act.stride(1),
        down_rank_out.stride(0),
        down_rank_out.stride(1),
        W=width,
        RANK_DOWN=down_rank,
        RANK_DOWN_PADDED=max(16, triton.next_power_of_2(down_rank)),
        NUM_SLICES=num_slices,
        ACT_RELU2=activation == "relu2",
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_W=block_size_w,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def invoke_fused_middle(
    bridge_gu: torch.Tensor,
    b_gate_up: torch.Tensor,
    base_gu: torch.Tensor,
    a_down: torch.Tensor,
    act: torch.Tensor,
    down_rank_out: torch.Tensor,
    routing: RouteView,
    *,
    activation: str,
    config: Mapping[str, int],
) -> None:
    num_slices, width, rank, down_rank = _validate_middle_call(
        activation=activation,
        base_gu=base_gu,
        act=act,
        routing=routing,
        bridge_gu=bridge_gu,
        b_gate_up=b_gate_up,
        a_down=a_down,
        down_rank_out=down_rank_out,
    )
    if routing.topk_ids.numel() == 0:
        return
    block_size_w = int(config["BLOCK_SIZE_W"])
    block_size_k = int(config["BLOCK_SIZE_K"])
    _require_dot_geometry(
        block_size_m=routing.block_size,
        block_size_w=block_size_w,
        block_size_k=block_size_k,
    )
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    _fused_middle_kernel[(num_m_blocks,)](
        bridge_gu,
        b_gate_up,
        base_gu,
        a_down,
        act,
        down_rank_out,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        routing.topk_ids.numel(),
        bridge_gu.stride(0),
        bridge_gu.stride(1),
        b_gate_up.stride(0),
        b_gate_up.stride(1),
        b_gate_up.stride(2),
        base_gu.stride(0),
        base_gu.stride(1),
        a_down.stride(0),
        a_down.stride(1),
        a_down.stride(2),
        act.stride(0),
        act.stride(1),
        down_rank_out.stride(0),
        down_rank_out.stride(1),
        W=width,
        RANK=rank,
        RANK_DOWN=down_rank,
        RANK_DOWN_PADDED=max(16, triton.next_power_of_2(down_rank)),
        NUM_SLICES=num_slices,
        ACT_RELU2=activation == "relu2",
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_W=block_size_w,
        BLOCK_SIZE_K=block_size_k,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


# (required, forbidden) tensor names per fusion. Forbidding what an arm
# claims to eliminate is the honest surface: a caller handing "b_act" a
# delta buffer — or expecting it to write down_rank_out — has wired the
# wrong arm, and silence here becomes unwritten-output corruption there.
_FUSION_TENSORS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "b_act": (
        ("bridge_gu", "b_gate_up"),
        ("gate_up_delta", "a_down", "down_rank_out"),
    ),
    "act_down_a": (
        ("gate_up_delta", "a_down", "down_rank_out"),
        ("bridge_gu", "b_gate_up"),
    ),
    "full": (
        ("bridge_gu", "b_gate_up", "a_down", "down_rank_out"),
        ("gate_up_delta",),
    ),
}


def run_fused_middle(
    spec: FusedMiddleSpec,
    *,
    base_gu: torch.Tensor,
    act: torch.Tensor,
    routing: RouteView,
    config: Mapping[str, int],
    bridge_gu: torch.Tensor | None = None,
    b_gate_up: torch.Tensor | None = None,
    gate_up_delta: torch.Tensor | None = None,
    a_down: torch.Tensor | None = None,
    down_rank_out: torch.Tensor | None = None,
) -> None:
    """Execute one fused-middle candidate FROM its spec — the spec IS the dispatch."""
    supplied = {
        "bridge_gu": bridge_gu,
        "b_gate_up": b_gate_up,
        "gate_up_delta": gate_up_delta,
        "a_down": a_down,
        "down_rank_out": down_rank_out,
    }
    required, forbidden = _FUSION_TENSORS[spec.fusion]
    for name in required:
        if supplied[name] is None:
            raise ValueError(f"fusion {spec.fusion!r} requires {name}")
    for name in forbidden:
        if supplied[name] is not None:
            raise ValueError(
                f"fusion {spec.fusion!r} does not consume {name}; refusing a "
                "tensor this arm claims to eliminate"
            )
    if spec.fusion == "b_act":
        invoke_fused_b_act(
            bridge_gu,
            b_gate_up,
            base_gu,
            act,
            routing,
            activation=spec.activation,
            config=config,
        )
    elif spec.fusion == "act_down_a":
        invoke_act_down_a(
            gate_up_delta,
            base_gu,
            a_down,
            act,
            down_rank_out,
            routing,
            activation=spec.activation,
            config=config,
        )
    elif spec.fusion == "full":
        invoke_fused_middle(
            bridge_gu,
            b_gate_up,
            base_gu,
            a_down,
            act,
            down_rank_out,
            routing,
            activation=spec.activation,
            config=config,
        )
    else:
        raise NotImplementedError(f"no executor for {spec.key()!r}")
