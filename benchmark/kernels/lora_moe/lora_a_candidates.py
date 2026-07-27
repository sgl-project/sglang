"""Step-3 LoRA-A schedule candidates (benchmark tier, plan §63.1 P2/P3).

Candidates live in the lab until gate 3 promotes a shortlist; the production
primitives in ``sgl_lora/bf16.py`` stay the correctness reference.  Families
here: INDEXED (raw-route) and DETERMINISTIC SPLIT-K (grouped).

Provenance: research-archive ``bench_indexed_shrink.py`` (4D direct
``[L, E, N, H]`` addressing over global ids) and
``algorithm_family_kernels.indexed_gemm`` (fused-group ``[L*E, N, K]``
addressing over an inline key).  Over a contiguous factor tensor the two
address identical bytes — ``group = adapter * E_f + factor`` IS the fused
key — so this port keeps ONE kernel and takes the key from
``routing.virtual_expert_ids_inline``, the single key/validity definition
(plan §29 R2; a re-inlined copy is how sentinel semantics silently diverge).

Contract (mirrors ``grouped_lora_a`` so call sites are parallel):

* input ``[T, H]`` token-major (gate/up site) or ``[P, K_dim]`` pair-major
  with ``pair_input=True`` (down site);
* weight ``[F, N, K_dim]`` — the flattened ``[L_cap * E_f, N, K_dim]``
  factor tensor;
* output ``[P, N]`` pair-major.  Rows at INVALID pairs are PRESERVED (never
  written) — strictly stronger than the grouped kernel's undefined-sentinel
  contract, and poison-testable; the pipeline's B-side zero-overwrite still
  owns sentinel destinations either way.

One program computes one ``(pair, BLOCK_N tile)`` result with a BN-by-BK
FP32 vector reduction — deliberately no ``tl.dot``: adjacent raw-route pairs
select unrelated (adapter, expert) weights, and measuring that no-plan
tradeoff against the aligned grouped schedule is the point of the arm.
Deterministic: serial K loop, FP32 accumulator, single store, no atomics.
"""

from __future__ import annotations

from typing import Mapping

import torch
import triton
import triton.language as tl

from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a
from sglang.srt.lora.sgl_lora.routing import RouteView, virtual_expert_ids_inline


def _validate_pair_gemm_call(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    *,
    pair_input: bool,
) -> None:
    """One compact contract check shared by every candidate invoke.

    Input rows (token- or pair-major per the site), the weight's
    (adapter x factor) group domain, the pair-major output, and device
    unity — a wrong tensor here is silent corruption in a masked kernel.
    """
    num_pairs = routing.topk_ids.numel()
    num_groups, n, k_dim = weight.shape
    expected_groups = routing.max_loras * routing.lora_experts_per_adapter
    if num_groups != expected_groups:
        raise ValueError(
            f"weight groups {num_groups} != max_loras * lora_experts_per_adapter "
            f"{expected_groups}"
        )
    if output.shape != (num_pairs, n):
        raise ValueError(f"output must have shape {(num_pairs, n)}")
    expected_rows = num_pairs if pair_input else routing.topk_ids.shape[0]
    if input.ndim != 2 or input.shape != (expected_rows, k_dim):
        raise ValueError(f"input must have shape {(expected_rows, k_dim)}")
    devices = {input.device, weight.device, output.device, routing.topk_ids.device}
    if len(devices) != 1:
        raise ValueError(f"tensors span devices {sorted(map(str, devices))}")


# The archive sweep grid, kept as the Step-3 tuning axes for this arm.
INDEXED_BLOCK_N = (8, 16, 32)
INDEXED_BLOCK_K = (32, 64, 128)
INDEXED_NUM_WARPS = (2, 4, 8)
INDEXED_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_N": 16,
    "BLOCK_SIZE_K": 64,
    "num_warps": 4,
    "num_stages": 3,
}


@triton.jit
def _indexed_lora_a_kernel(
    x_ptr,
    weight_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    lora_expert_map_ptr,
    output_ptr,
    num_pairs,
    routed_expert_id_bound,
    stride_xm,
    stride_xk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    N: tl.constexpr,
    K_DIM: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    INPUT_PAIR_MAJOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    # The ONE key/validity definition (R2). group == fused key by
    # construction: adapter * LORA_EXPERTS_PER_ADAPTER + factor.
    key = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        lora_expert_map_ptr,
        pair,
        pair < num_pairs,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=USE_LORA_EXPERT_MAP,
        SHARED_OUTER=SHARED_OUTER,
    )
    valid = key != -1
    # int64 addresses: group * stride_wg overflows int32 at large L*E*N*K.
    group = tl.maximum(key, 0).to(tl.int64)
    pair64 = pair.to(tl.int64)
    x_row = pair64 if INPUT_PAIR_MAJOR else pair64 // TOP_K

    offs_n = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    n_mask = offs_n < N
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K_DIM, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
        k_mask = offs_k < K_DIM
        x = tl.load(
            x_ptr + x_row * stride_xm + offs_k * stride_xk,
            mask=valid & k_mask,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + group * stride_wg
            + offs_n[:, None] * stride_wn
            + offs_k[None, :] * stride_wk,
            mask=valid & n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accumulator += tl.sum(weight.to(tl.float32) * x[None, :].to(tl.float32), axis=1)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    tl.store(
        output_ptr + pair64 * stride_om + offs_n * stride_on,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=valid & n_mask,
    )


def invoke_indexed_lora_a(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    *,
    config: Mapping[str, int],
    pair_input: bool = False,
) -> None:
    """Launch the indexed candidate off a RouteView's SOURCES.

    Any view works — the kernel reads only ``topk_ids`` / ``token_slots`` /
    the LoRA expert map, which every view carries; an honest route-inclusive
    comparison still requests ``ROUTE_RAW`` so the arm is charged exactly
    the route work it causes (none).
    """
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return
    _validate_pair_gemm_call(input, weight, output, routing, pair_input=pair_input)
    n = weight.shape[1]
    k_dim = weight.shape[2]

    from sglang.kernels.jit.utils import is_arch_support_pdl

    lora_expert_map = routing.lora_expert_map
    use_map = lora_expert_map is not None
    shared_outer_local_expert_count = routing.shared_outer_local_expert_count
    use_pdl = is_arch_support_pdl()
    grid = (num_pairs, triton.cdiv(n, config["BLOCK_SIZE_N"]))
    _indexed_lora_a_kernel[grid](
        input,
        weight,
        routing.topk_ids,
        routing.token_slots,
        routing.topk_ids if not use_map else lora_expert_map,
        output,
        num_pairs,
        (
            shared_outer_local_expert_count
            if shared_outer_local_expert_count is not None
            else (0 if not use_map else lora_expert_map.numel())
        ),
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        N=n,
        K_DIM=k_dim,
        LORA_EXPERTS_PER_ADAPTER=routing.lora_experts_per_adapter,
        MAX_LORAS=routing.max_loras,
        TOP_K=routing.topk_ids.shape[1],
        USE_LORA_EXPERT_MAP=use_map,
        SHARED_OUTER=shared_outer_local_expert_count is not None,
        INPUT_PAIR_MAJOR=pair_input,
        BLOCK_N=config["BLOCK_SIZE_N"],
        BLOCK_K=config["BLOCK_SIZE_K"],
        USE_PDL=use_pdl,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        **({"launch_pdl": True} if use_pdl else {}),
    )


# ---------------------------------------------------------------------------
# DETERMINISTIC SPLIT-K (grouped) — the §14 family no archive code implements:
# every archive split-K accumulates with bf16 tl.atomic_add (non-deterministic
# by construction; §14 demotes it to diagnostic-only).  This candidate splits
# the K reduction across programs into an FP32 WORKSPACE and sums the partials
# in a FIXED serial order, so it meets the same bitwise replay bar as the
# single-K kernels.  Motivation is decode occupancy: the aligned grouped grid
# at small P is a handful of programs while the K loop (H/BLOCK_K iters)
# dominates this skinny-N GEMM.
#
# Split layout is STRIDED: split s takes every SPLIT_K-th BLOCK_K tile
# (k = s*BLOCK_K, (s+SPLIT_K)*BLOCK_K, ...), which balances work with no
# chunk arithmetic and keeps each partial's internal order deterministic.
# The reduce walks splits 0..SPLIT_K-1 serially and casts ONCE to the output
# dtype.  Sentinel M-blocks are skipped by both kernels, so output rows at
# sentinel pairs stay undefined — the same contract as grouped_lora_a (the
# paired B primitive overwrites its destinations).
# ---------------------------------------------------------------------------

SPLIT_K_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
    "SPLIT_K": 0,  # 0 = use split_k_heuristic at launch
}


def split_k_heuristic(
    *, n: int, k_dim: int, capacity: int, block_m: int, block_n: int, block_k: int
) -> int:
    """Rank-tiered occupancy fill, ported from the archive shrink (PR #26899).

    Skinnier ranks want more splits (their output tile carries less work);
    the target tiers came from an offline per-M B200 sweep and land within
    ~5% of per-shape tuned optima across the decode regime THERE — on this
    campaign's devices they are a PRIOR to sweep against, not a result.
    Static per shape (capacity, not the live pair count), hence graph-safe.
    """
    base_grid = -(-capacity // block_m) * -(-n // block_n)
    target = 512 if n <= 16 else 384 if n <= 32 else 256
    max_split = max(1, k_dim // block_k)
    return max(1, min(-(-target // base_grid), max_split, 8))


@triton.jit
def _split_k_lora_a_partial_kernel(
    input_ptr,
    weight_ptr,
    workspace_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_input_rows,
    num_pairs,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wk,
    stride_ws,
    stride_wm,
    stride_wsn,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PAIR_INPUT: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    # Same grouped-M swizzle as the single-K kernel (second S3 review: a
    # scheduling difference between the arms would masquerade as a split-K
    # effect exactly where the N-tile count changes with rank).
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    programs_per_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_M_BLOCKS - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % programs_per_group) % group_size_m)
    pid_n = (pid % programs_per_group) // group_size_m
    if pid_m * BLOCK_SIZE_M >= tl.load(num_pairs_post_padded_ptr):
        return
    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    virtual_expert_id = tl.load(block_virtual_expert_ids_ptr + pid_m).to(tl.int64)
    if virtual_expert_id == -1:
        return
    input_rows = pair_ids if PAIR_INPUT else pair_ids // TOP_K
    input_mask = pair_mask & (input_rows < num_input_rows)

    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_begin in range(split * BLOCK_SIZE_K, K, SPLIT_K * BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < K
        lhs = tl.load(
            input_ptr
            + input_rows[:, None] * stride_im
            + k_offsets[None, :] * stride_ik,
            mask=input_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + virtual_expert_id * stride_we
            + n_offsets[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.float32)

    tl.store(
        workspace_ptr
        + split * stride_ws
        + pair_ids[:, None] * stride_wm
        + n_offsets[None, :] * stride_wsn,
        accumulator,
        mask=pair_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _split_k_lora_a_reduce_kernel(
    workspace_ptr,
    output_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs_post_padded_ptr,
    num_pairs,
    stride_ws,
    stride_wm,
    stride_wsn,
    stride_om,
    stride_on,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_m * BLOCK_SIZE_M >= tl.load(num_pairs_post_padded_ptr):
        return
    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    if tl.load(block_virtual_expert_ids_ptr + pid_m) == -1:
        return
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < N
    mask = pair_mask[:, None] & n_mask[None, :]
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # FIXED order: the loop bound is a constexpr and the sum is serial, so
    # the reduction tree is identical on every launch — the determinism bar.
    for split in range(SPLIT_K):
        accumulator += tl.load(
            workspace_ptr
            + split * stride_ws
            + pair_ids[:, None] * stride_wm
            + n_offsets[None, :] * stride_wsn,
            mask=mask,
            other=0.0,
        )
    tl.store(
        output_ptr + pair_ids[:, None] * stride_om + n_offsets[None, :] * stride_on,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def split_k_workspace(output: torch.Tensor, *, split_k: int) -> torch.Tensor:
    """FP32 partials buffer for `invoke_split_k_lora_a` (caller-owned)."""
    return torch.empty(
        (split_k, *output.shape), dtype=torch.float32, device=output.device
    )


def resolve_split_k(
    weight: torch.Tensor, routing: RouteView, config: Mapping[str, int]
) -> int:
    """The SPLIT_K this launch will use (config override or heuristic)."""
    configured = int(config.get("SPLIT_K", 0))
    if configured:
        return configured
    return split_k_heuristic(
        n=weight.shape[1],
        k_dim=weight.shape[2],
        capacity=routing.sorted_pair_ids.numel(),
        block_m=routing.block_size,
        block_n=int(config["BLOCK_SIZE_N"]),
        block_k=int(config["BLOCK_SIZE_K"]),
    )


def invoke_split_k_lora_a(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    *,
    config: Mapping[str, int],
    pair_input: bool = False,
    workspace: torch.Tensor | None = None,
) -> None:
    """Deterministic split-K grouped LoRA-A: FP32 partials, fixed-order sum.

    ``workspace`` (from :func:`split_k_workspace`, sized for
    :func:`resolve_split_k`) lets a timing thunk hoist the allocation; when
    None one is allocated per call.  SPLIT_K == 1 still runs both kernels —
    the uniform two-launch shape keeps the arm's overhead visible instead of
    silently becoming the single-K kernel.
    """
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return
    _validate_pair_gemm_call(input, weight, output, routing, pair_input=pair_input)
    split_k = resolve_split_k(weight, routing, config)
    if split_k < 1:
        raise ValueError(f"resolved split factor must be positive, got {split_k}")
    if workspace is None:
        workspace = split_k_workspace(output, split_k=split_k)
    if workspace.shape != (split_k, *output.shape):
        raise ValueError(
            f"workspace shape {tuple(workspace.shape)} != "
            f"{(split_k, *output.shape)} (resolve_split_k first)"
        )
    if workspace.dtype != torch.float32:
        raise ValueError("split-K workspace must be FP32")
    plan_tensors = (
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
    )
    devices = {workspace.device, output.device, *(t.device for t in plan_tensors)}
    if len(devices) != 1:
        raise ValueError(
            f"workspace/plan tensors span devices {sorted(map(str, devices))}"
        )
    n = weight.shape[1]
    block_size_n = int(config["BLOCK_SIZE_N"])
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    num_n_blocks = triton.cdiv(n, block_size_n)
    base_grid = num_m_blocks * num_n_blocks
    _split_k_lora_a_partial_kernel[(base_grid, split_k)](
        input,
        weight,
        workspace,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        input.shape[0],
        num_pairs,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        workspace.stride(0),
        workspace.stride(1),
        workspace.stride(2),
        TOP_K=routing.topk_ids.shape[1],
        N=n,
        K=weight.shape[2],
        PAIR_INPUT=pair_input,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        GROUP_SIZE_M=int(config.get("GROUP_SIZE_M", 8)),
        SPLIT_K=split_k,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
    _split_k_lora_a_reduce_kernel[(base_grid,)](
        workspace,
        output,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        num_pairs,
        workspace.stride(0),
        workspace.stride(1),
        workspace.stride(2),
        output.stride(0),
        output.stride(1),
        N=n,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        SPLIT_K=split_k,
        num_warps=4,
        num_stages=2,
    )


def run_lora_a(
    spec: LoraAExecutionSpec,
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    config: Mapping[str, int],
    workspace: torch.Tensor | None = None,
    cutedsl_plan=None,
) -> None:
    """Execute one A candidate FROM its spec — the spec IS the dispatch.

    Second S3 review: string-keyed dispatch with a fall-through arm let a
    recorded spec and the executed kernel diverge silently.  This is the
    one executor every driver goes through: exhaustive on the (ownership,
    reduction, implementation) combinations that exist, raising on
    everything else.  ``pair_input`` derives from the spec's site, and the
    token-dedup shared form runs its own entry points in
    ``lora_a_shared`` (a different route domain and bridge shape, not a
    kernel variant of this signature).
    """
    if spec.shared_handling != "repeated_pairs":
        raise NotImplementedError(
            "token-dedup runs through lora_a_shared (T-domain plan and "
            "token-major bridge), not this pair-domain executor"
        )
    if spec.implementation == "cutedsl":
        # P5 composite (lora_a_cutedsl): masked-row staging + the S2 masked
        # grouped GEMM + pair-major scatter-back. The plan carries the
        # compiled kernels and the fused-id dispatch for THIS fixture; a
        # declared cutedsl spec without one is a driver bug, not a fallback.
        if cutedsl_plan is None:
            raise ValueError(
                f"spec {spec.key()!r} declares implementation=cutedsl but no "
                "cutedsl_plan was supplied (build_cutedsl_lora_a_plan)"
            )
        if spec.ownership != "grouped" or spec.reduction != "whole_rank":
            raise NotImplementedError(
                "the CuTeDSL arm is the masked GROUPED whole-rank composite; "
                f"got ownership={spec.ownership!r} reduction={spec.reduction!r}"
            )
        # Seventh/eighth S3 reviews: the plan must be bound to THIS
        # invocation's SITE, weights, and route source tensors.
        cutedsl_plan.require_binding(spec.site, weight, routing)
        if spec.site == "gate_up":
            cutedsl_plan.run_gate_up(input, output)
        else:
            cutedsl_plan.run_down(input, output)
        return
    pair_input = spec.site == "down"
    if spec.ownership == "grouped" and spec.reduction == "whole_rank":
        grouped_lora_a(
            input, weight, output, routing, config=config, pair_input=pair_input
        )
    elif spec.ownership == "grouped" and spec.reduction == "deterministic_split_k":
        invoke_split_k_lora_a(
            input,
            weight,
            output,
            routing,
            config=config,
            pair_input=pair_input,
            workspace=workspace,
        )
    elif spec.ownership == "indexed" and spec.reduction == "whole_rank":
        invoke_indexed_lora_a(
            input, weight, output, routing, config=config, pair_input=pair_input
        )
    else:
        raise NotImplementedError(
            f"no executor for ownership={spec.ownership!r} with "
            f"reduction={spec.reduction!r}"
        )
