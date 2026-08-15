"""Selector-free BF16 LoRA-A candidate kernels for the SGL MoE backend.

This module deliberately carries no selection logic.  It contains only
the families the production config resolver can reach:

* aligned grouped A, the general reference/winner (implemented in ``bf16``);
* raw-route indexed A, retained for the evidence-qualified down-A
  small-decode composition; and
* token-deduplicated shared-outer grouped A.

The caller owns route selection, launch-config selection, workspace lifetime,
and composition with B.  Indexed A consumes a raw ``RouteView`` and has no PDL
annotations. Grouped A exposes an explicit producer flag used only by legal,
same-stream A-to-B execution-plan twins; it is never an implicit
device-dependent behavior inside the math primitive.

Port provenance (the source launchers remain benchmark-only controls):

* ``grouped_lora_a`` wraps ``MoE LoRA.bf16.grouped_lora_a``;
* ``indexed_lora_a`` mirrors
  ``benchmark.kernels.lora_moe.lora_a_candidates.invoke_indexed_lora_a``; and
* ``token_dedup_grouped_lora_a`` is the A half of
  ``lora_a_shared.shared_gate_up_a_token_dedup``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.bf16 import grouped_lora_a
from sglang.srt.lora.moe.routing import RouteView, virtual_expert_ids_inline

if TYPE_CHECKING:
    from sglang.srt.lora.moe.execution_plan import LoraASpec


def _spec_value(spec: object, field: str) -> str:
    """Read a string/Enum field without coupling kernels to selector code."""
    value = getattr(spec, field, None)
    if value is None:
        raise ValueError(f"LoRA-A execution spec is missing {field!r}")
    return str(getattr(value, "value", value))


def _validate_pair_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    *,
    pair_input: bool,
) -> None:
    num_pairs = routing.topk_ids.numel()
    if weight.ndim != 3:
        raise ValueError(f"weight must be 3D, got shape {tuple(weight.shape)}")
    num_groups, width, input_width = weight.shape
    expected_groups = routing.max_loras * routing.lora_experts_per_adapter
    if num_groups != expected_groups:
        raise ValueError(
            f"weight groups {num_groups} != max_loras * "
            f"lora_experts_per_adapter {expected_groups}"
        )
    expected_rows = num_pairs if pair_input else routing.topk_ids.shape[0]
    if input.ndim != 2 or input.shape != (expected_rows, input_width):
        raise ValueError(f"input must have shape {(expected_rows, input_width)}")
    if output.ndim != 2 or output.shape != (num_pairs, width):
        raise ValueError(f"output must have shape {(num_pairs, width)}")
    devices = {
        input.device,
        weight.device,
        output.device,
        routing.topk_ids.device,
    }
    if len(devices) != 1:
        raise ValueError(f"tensors span devices {sorted(map(str, devices))}")


@triton.jit
def _indexed_lora_a_kernel(
    input_ptr,
    weight_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    lora_expert_map_ptr,
    output_ptr,
    num_pairs,
    routed_expert_id_bound,
    stride_im,
    stride_ik,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    N: tl.constexpr,
    K: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_LORA_EXPERT_MAP: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    PAIR_INPUT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One deterministic vector reduction per raw-route pair and N tile."""
    pair_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    key = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_slots_ptr,
        lora_expert_map_ptr,
        pair_id,
        pair_id < num_pairs,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        USE_LORA_EXPERT_MAP=USE_LORA_EXPERT_MAP,
        SHARED_OUTER=SHARED_OUTER,
    )
    valid = key != -1
    group = tl.maximum(key, 0).to(tl.int64)
    pair64 = pair_id.to(tl.int64)
    input_row = pair64 if PAIR_INPUT else pair64 // TOP_K

    n_offsets = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(
        tl.int64
    )
    n_mask = n_offsets < N
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_begin in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < K
        lhs = tl.load(
            input_ptr + input_row * stride_im + k_offsets * stride_ik,
            mask=valid & k_mask,
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + group * stride_wg
            + n_offsets[:, None] * stride_wn
            + k_offsets[None, :] * stride_wk,
            mask=valid & n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accumulator += tl.sum(rhs.to(tl.float32) * lhs[None, :].to(tl.float32), axis=1)

    # A preserves invalid rows.  B owns and zero-fills every consumed
    # destination cell, so an indexed A need not write sentinel pairs.
    tl.store(
        output_ptr + pair64 * stride_om + n_offsets * stride_on,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=valid & n_mask,
    )


def indexed_lora_a(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    *,
    config: Mapping[str, int],
    pair_input: bool = False,
) -> None:
    """Execute raw-route indexed A.

    This launcher intentionally contains no ``launch_pdl`` or GDC operations.
    A caller choosing this family should request ``ROUTE_RAW`` so no aligned
    plan is built merely for another site.
    """
    num_pairs = routing.topk_ids.numel()
    if num_pairs == 0:
        return
    _validate_pair_gemm(input, weight, output, routing, pair_input=pair_input)
    use_map = routing.lora_expert_map is not None
    shared_outer = routing.shared_outer_local_expert_count is not None
    routed_bound = (
        routing.shared_outer_local_expert_count
        if shared_outer
        else (routing.lora_expert_map.numel() if use_map else 0)
    )
    map_arg = routing.lora_expert_map if use_map else routing.topk_ids
    block_size_n = int(config["BLOCK_SIZE_N"])
    _indexed_lora_a_kernel[(num_pairs, triton.cdiv(weight.shape[1], block_size_n))](
        input,
        weight,
        routing.topk_ids,
        routing.token_slots,
        map_arg,
        output,
        num_pairs,
        routed_bound,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        N=weight.shape[1],
        K=weight.shape[2],
        LORA_EXPERTS_PER_ADAPTER=routing.lora_experts_per_adapter,
        MAX_LORAS=routing.max_loras,
        TOP_K=routing.topk_ids.shape[1],
        USE_LORA_EXPERT_MAP=use_map,
        SHARED_OUTER=shared_outer,
        PAIR_INPUT=pair_input,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def token_dedup_grouped_lora_a(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    token_routing: RouteView,
    *,
    config: Mapping[str, int],
    produce_pdl: bool = False,
) -> None:
    """Shared-outer A over a T-domain adapter plan.

    ``token_routing`` has top-k one, so the ordinary grouped primitive writes
    one bridge row per token instead of repeating it for every routed expert.
    Plan construction belongs to routing, not this math module.
    """
    if token_routing.topk_ids.shape[1] != 1:
        raise ValueError(
            "token-deduplicated grouped A requires a T-domain top-k-1 route"
        )
    if token_routing.lora_experts_per_adapter != 1:
        raise ValueError(
            "token-deduplicated grouped A requires one shared factor per adapter"
        )
    _validate_pair_gemm(input, weight, output, token_routing, pair_input=False)
    grouped_lora_a(
        input,
        weight,
        output,
        token_routing,
        config=config,
        produce_pdl=produce_pdl,
    )


def run_lora_a(
    spec: LoraASpec,
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    config: Mapping[str, int],
    input_row_map: torch.Tensor | None = None,
    produce_pdl: bool = False,
) -> torch.Tensor:
    """Execute exactly the A family named by an execution-plan spec.

    No fallback or selector lives here. Every family writes and returns the
    caller-owned ``output``.
    """
    family = _spec_value(spec, "family")
    site = _spec_value(spec, "site")
    pair_input = site == "down"
    if site not in ("gate_up", "down"):
        raise ValueError(f"unknown LoRA-A site {site!r}")
    if input_row_map is not None and not (family == "grouped" and site == "down"):
        raise ValueError(
            "a provider input_row_map is supported only by standalone grouped down-A"
        )

    if family in ("grouped", "token_dedup_grouped"):
        if family == "token_dedup_grouped":
            if pair_input:
                raise ValueError("token-deduplicated A exists only at gate_up")
            token_dedup_grouped_lora_a(
                input,
                weight,
                output,
                routing,
                config=config,
                produce_pdl=produce_pdl,
            )
        else:
            grouped_lora_a(
                input,
                weight,
                output,
                routing,
                config=config,
                pair_input=pair_input,
                input_row_map=input_row_map,
                produce_pdl=produce_pdl,
            )
        return output

    if produce_pdl:
        raise ValueError(
            f"{family} A has no qualified programmatic-dependent-launch producer"
        )

    if family == "indexed":
        indexed_lora_a(
            input,
            weight,
            output,
            routing,
            config=config,
            pair_input=pair_input,
        )
        return output

    raise NotImplementedError(f"no production LoRA-A executor for {family!r}")
