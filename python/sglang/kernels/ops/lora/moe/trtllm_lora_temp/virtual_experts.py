"""
LoRA Virtual Experts Triton Ops.
"""

import functools
from typing import Any

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.lora.dense.trtllm_lora_temp.kernel_utils import (
    get_pdl_launch_metadata,
)
from sglang.kernels.ops.lora.moe.virtual_experts import _align_block_size_large
from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs


@triton.jit
def _fused_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    virtual_topk_ids_ptr,
    token_lora_mask_ptr,
    adapter_enabled_ptr,
    stride_enabled,
    num_experts_for_weight: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    M,
    top_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    local_expert_offset: tl.constexpr,
    local_num_experts: tl.constexpr,
    EP_LOCAL: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    HAS_ADAPTER_ENABLED: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """
    Map active slots and valid base experts to virtual weight indices.

    Invalid slots or base experts map to -1. Shared weights accept any
    nonnegative base expert and use the adapter slot as the weight index.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = M * top_k
    valid = offs < total

    # GDC wait: topk_ids/token_lora_mapping come from the immediately
    # preceding routing kernels; wait before consuming them.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    m = offs // top_k
    # k = offs % top_k  # not needed directly

    lora_id = tl.load(token_lora_mapping_ptr + m, mask=valid, other=0)
    mask_val = (lora_id >= 0) & (lora_id < NUM_SLOTS)
    if HAS_ADAPTER_ENABLED:
        if NUM_SLOTS == 1:
            mask_val = mask_val & (tl.load(adapter_enabled_ptr) != 0)
        else:
            mask_val = mask_val & (
                tl.load(
                    adapter_enabled_ptr + lora_id * stride_enabled,
                    mask=valid & mask_val,
                    other=0,
                )
                != 0
            )
    safe_lora = tl.where(mask_val, lora_id, 0)

    base = tl.load(topk_ids_ptr + offs, mask=valid, other=0)
    if EP_LOCAL:
        # EP: drop experts this rank does not own to the -1 sentinel (handled just
        # below); owned experts keep their GLOBAL id so the global contiguous weight
        # buffer indexes with a single stride and the merged-weight reshape stays a
        # free view. Fused in here -> no separate where()/elementwise launch.
        owned = (base >= local_expert_offset) & (
            base < local_expert_offset + local_num_experts
        )
        base = tl.where(owned, base, -1)
    valid_base = base >= 0
    if not SHARED_OUTER:
        valid_base = valid_base & (base < num_experts_for_weight)
    expert = 0 if SHARED_OUTER else tl.where(valid_base, base, 0)
    shifted = expert + safe_lora * num_experts_for_weight
    # All align providers use one sentinel bucket; borrowed route IDs stay intact.
    result = tl.where(mask_val & valid_base, shifted, -1)
    tl.store(virtual_topk_ids_ptr + offs, result, mask=valid)

    # Write mask once per row (at first k position)
    k = offs % top_k
    is_first_k = k == 0
    tl.store(token_lora_mask_ptr + m, mask_val, mask=valid & is_first_k)

    # All work is done; hint the runtime to launch the dependent kernel.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _fused_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    shared_outer: bool,
    max_loras: int,
    local_expert_offset: int = 0,
    local_num_experts: int | None = None,
    adapter_enabled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns virtual topk_ids, token_lora_mask, and virtual_num_experts.

    `local_num_experts` (< `num_experts`) enables the EP mask on the per-expert
    path: non-owned `topk_ids` slots are dropped to the -1 sentinel in-kernel.
    The shared-outer path drops negative routes but does not apply the local
    expert range (its weights are shared by every owned expert).
    """
    M, top_k = topk_ids.shape
    device = topk_ids.device

    num_experts_for_weight = 1 if shared_outer else num_experts
    ep_local = (
        not shared_outer
        and local_num_experts is not None
        and local_num_experts < num_experts
    )

    virtual_topk_ids = torch.empty_like(topk_ids)
    token_lora_mask = torch.empty(M, dtype=torch.bool, device=device)
    if M * top_k == 0:
        return virtual_topk_ids, token_lora_mask, num_experts_for_weight * max_loras

    BLOCK_SIZE = 1024
    grid = ((M * top_k + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _fused_virtual_topk_ids_kernel[grid](
        topk_ids,
        token_lora_mapping,
        virtual_topk_ids,
        token_lora_mask,
        adapter_enabled,
        adapter_enabled.stride(0) if adapter_enabled is not None else 0,
        num_experts_for_weight,
        shared_outer,
        M,
        top_k,
        BLOCK_SIZE,
        local_expert_offset,
        local_num_experts if local_num_experts is not None else 0,
        ep_local,
        NUM_SLOTS=max_loras,
        HAS_ADAPTER_ENABLED=adapter_enabled is not None,
        ENABLE_PDL=enable_pdl,
        **pdl_kwargs,
    )

    virtual_num_experts = num_experts_for_weight * max_loras
    return virtual_topk_ids, token_lora_mask, virtual_num_experts


@triton.jit
def _moe_lora_shrink_splitk_kernel(
    # Pointers
    a_ptr,  # type: ignore  # [num_tokens, K]
    b_ptr,  # type: ignore  # [num_virtual_experts, N, K]
    c_ptr,  # type: ignore  # [num_tokens * top_k, N]  (pre-zeroed when SPLIT_K > 1)
    sorted_token_ids_ptr,  # type: ignore
    expert_ids_ptr,  # type: ignore
    num_tokens_post_padded_ptr,  # type: ignore
    # Dimensions
    N,  # type: ignore
    K,  # type: ignore
    num_valid_tokens,  # type: ignore
    # Strides
    stride_am,  # type: ignore
    stride_ak,  # type: ignore
    stride_be,  # type: ignore
    stride_bn,  # type: ignore
    stride_bk,  # type: ignore
    stride_cm,  # type: ignore
    stride_cn,  # type: ignore
    # Constexprs
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Split-K grouped GEMM for the LoRA A (shrink) stage with few virtual experts."""
    pid = tl.program_id(0)
    pid_sk = pid % SPLIT_K
    pid_mn = pid // SPLIT_K

    # GDC wait: the routing buffers (sorted_token_ids/expert_ids/
    # num_tokens_post_padded) come from routing preparation; wait before
    # consuming them.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if pid_mn >= num_pid_m * num_pid_n:
        return

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    # Token routing (same pattern as fused_moe_triton_kernels)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        return

    # Pointers
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Accumulate
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    for k in range(0, grid_k):
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        k_mask = offs_k[:, None] < k_remaining
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b.to(a.dtype))
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    # All input reads are done; hint the runtime to launch the dependent kernel.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # Write output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


def _invoke_moe_lora_shrink_splitk(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: dict[str, Any],
) -> None:
    """Launch split-K shrink kernel for LoRA A with few virtual experts."""
    N = weight.shape[1]
    K = weight.shape[2]
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(N))
    BLOCK_SIZE_K = 256
    GROUP_SIZE_M = config.get("GROUP_SIZE_M", 1)

    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], BLOCK_SIZE_M)
    num_n_blocks = triton.cdiv(N, BLOCK_SIZE_N)
    base_grid = num_m_blocks * num_n_blocks
    # Single source of truth shared with the caller's zero-intermediate decision:
    # split-K accumulation REQUIRES a pre-zeroed output, so the predicted and
    # launched SPLIT_K must never diverge.
    SPLIT_K = _get_moe_lora_shrink_split_k(weight, sorted_token_ids, config)

    grid = (SPLIT_K * base_grid,)

    enable_pdl, pdl_kwargs = get_pdl_launch_metadata()
    _moe_lora_shrink_splitk_kernel[grid](
        hidden_states,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        topk_ids.numel(),
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=SPLIT_K,
        ENABLE_PDL=enable_pdl,
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 4),
        **pdl_kwargs,
    )


def _get_moe_lora_shrink_split_k(
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    config: dict[str, Any],
) -> int:
    """Choose split-K from rank and available occupancy.

    Skinny output ranks benefit from more K splits because each output tile
    carries less work. Block sizes must mirror _invoke_moe_lora_shrink_splitk.

    """
    N = weight.shape[1]
    K = weight.shape[2]
    block_size_m = config["BLOCK_SIZE_M"]
    block_size_n = min(128, triton.next_power_of_2(N))
    block_size_k = 256
    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], block_size_m)
    base_grid = num_m_blocks * triton.cdiv(N, block_size_n)
    target = 512 if N <= 16 else 384 if N <= 32 else 256
    max_split_k = max(1, K // block_size_k)
    return max(1, min(triton.cdiv(target, base_grid), max_split_k, 8))


# Rank-specialized LoRA-B expand kernel lives in srt/lora/trtllm_lora_temp/.
# Re-export so existing call sites (and any external imports) keep working.
from sglang.srt.lora.trtllm_lora_temp.specialized_expand import (  # noqa: E402,F401
    _invoke_moe_lora_expand_add,
    _moe_lora_expand_add_kernel,
)


def _merged_experts_fused_moe_lora_add_fake(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
) -> None:
    return


def _get_stage_config(
    weight_shape: tuple[int, ...],
    stage_top_k: int,
    dtype: torch.dtype,
    num_tokens: int,
) -> dict[str, Any]:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
        try_get_optimal_moe_config,
    )

    config_dtype = get_config_dtype_str(dtype=dtype)
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        weight_shape,
        weight_shape,
        stage_top_k,
        config_dtype,
    )
    try:
        cfg = get_config_func(num_tokens)
    except ValueError:
        K_dim = weight_shape[2]
        N_dim = weight_shape[1]
        if K_dim >= 1024:
            default_block_k = 256
        elif K_dim >= 64:
            default_block_k = 64
        else:
            default_block_k = max(16, K_dim)
        cfg = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": min(64, max(16, N_dim)),
            "BLOCK_SIZE_K": min(default_block_k, max(16, K_dim)),
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4,
        }
    return cfg


def _get_shrink_stage_config(
    weight_shape: tuple[int, ...], M: int, reused_block_size: int | None = None
) -> dict[str, Any]:
    """Choose the shrink kernel config and the matching routing block size."""
    N = weight_shape[1]
    if M < 512:
        config = {
            "BLOCK_SIZE_M": 16,
            "num_warps": 4 if (M <= 4 or N >= 32) else 2,
            "num_stages": 3,
        }
    else:
        config = {"BLOCK_SIZE_M": 32, "num_warps": 2, "num_stages": 2}
    if lora_envs.SGLANG_OPT_LORA_SHRINK_TUNE.get():
        config = {"BLOCK_SIZE_M": 16, "num_warps": 4, "num_stages": 4}
    if reused_block_size is not None:
        config["BLOCK_SIZE_M"] = reused_block_size
    return config


def _routing_cache_key(
    num_experts, shared_outer, block_size, include_inactive_routes, adapter_enabled
):
    return (
        num_experts,
        shared_outer,
        block_size,
        include_inactive_routes,
        id(adapter_enabled),
    )


def _align_virtual_topk_ids(
    virtual_topk_ids, block_size, virtual_num_experts, populated_buckets
):
    if virtual_num_experts < 1024:
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size as native_moe_align_block_size,
        )

        sorted_ids, expert_ids, count = native_moe_align_block_size(
            virtual_topk_ids, block_size, virtual_num_experts
        )
    else:
        sorted_ids, expert_ids, count = _align_block_size_large(
            virtual_topk_ids, block_size, virtual_num_experts
        )
    num_tokens = virtual_topk_ids.numel()
    max_nonempty = min(num_tokens, populated_buckets)
    tight_padded = (
        triton.cdiv(num_tokens + max_nonempty * (block_size - 1), block_size)
        * block_size
    )
    return sorted_ids[:tight_padded], expert_ids[: tight_padded // block_size], count


def _prewarm_per_expert_lora_routes(
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    routing_cache: dict,
    *,
    shrink_lora_b: torch.Tensor,
    adapter_enabled: torch.Tensor | None = None,
    include_inactive_routes: bool = False,
    local_expert_offset: int = 0,
    local_num_experts: int | None = None,
) -> bool:
    """Prepare paired per-expert stages on main and release raw IDs before returning.

    The aligned cache retains only tables and their shared token mask. Global-ID
    EP routing keeps the existing provider admission and staged fallback.
    """
    if lora_a.shape[:2] != lora_b.shape[:2]:
        return False
    slots, num_experts = lora_a.shape[:2]
    if local_num_experts is not None and local_num_experts < num_experts:
        return False
    if adapter_enabled is not None:
        assert adapter_enabled.shape == (slots,)
        assert adapter_enabled.dtype == torch.int32
        assert adapter_enabled.device == topk_ids.device
    if topk_ids.numel() == 0:
        return True
    a_shape = (slots * num_experts, lora_a.shape[2], lora_a.shape[3])
    b_shape = (slots * num_experts, lora_b.shape[2], lora_b.shape[3])
    tokens = token_lora_mapping.shape[0]
    b_config = _get_stage_config(b_shape, 1, hidden_states.dtype, tokens)
    reuse_b_block = (
        tokens >= 512 and lora_envs.SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE.get()
    )
    reused_block_size = None
    if reuse_b_block:
        shrink_b_shape = (
            shrink_lora_b.shape[0] * shrink_lora_b.shape[1],
            shrink_lora_b.shape[2],
            shrink_lora_b.shape[3],
        )
        reused_block_size = _get_stage_config(
            shrink_b_shape, 1, hidden_states.dtype, tokens
        )["BLOCK_SIZE_M"]
    a_config = _get_shrink_stage_config(a_shape, tokens, reused_block_size)
    missing = {}
    for block_size in (b_config["BLOCK_SIZE_M"], a_config["BLOCK_SIZE_M"]):
        key = _routing_cache_key(
            num_experts, False, block_size, include_inactive_routes, adapter_enabled
        )
        if key not in routing_cache:
            missing[key] = block_size
    if not missing:
        return True
    virtual_ids, token_mask, virtual_experts = _fused_virtual_topk_ids(
        topk_ids,
        token_lora_mapping,
        num_experts,
        False,
        slots,
        local_expert_offset,
        local_num_experts,
        adapter_enabled=adapter_enabled,
    )
    for key, block_size in missing.items():
        sorted_ids, expert_ids, count = _align_virtual_topk_ids(
            virtual_ids, block_size, virtual_experts, virtual_experts
        )
        routing_cache[key] = (sorted_ids, expert_ids, count, token_mask)
    # Only aligned tables/masks survive the main-stream preparation scope.
    del virtual_ids
    return True


def _merged_experts_fused_moe_lora_add_impl(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    routing_cache: dict | None = None,
    fuse_add_to_output: bool = True,
    fuse_sum_all_reduce: bool = False,
    use_direct_expand_add: bool = False,
    local_expert_offset: int = 0,
    local_num_experts: int | None = None,
    stage: str = "all",
    intermediate_buffer: torch.Tensor | None = None,
    expand_wait_event: "torch.cuda.Event | None" = None,
    broadcast_intermediate: bool = False,
    prewarm_a_routing: bool = True,
    prewarm_b_routing: bool = True,
    zero_intermediate: bool = False,
    include_inactive_routes: bool = False,
    adapter_enabled: torch.Tensor | None = None,
) -> "torch.Tensor | None":
    """
    1. Prepare virtual expert routing metadata from topk_ids + token_lora_mapping * num_experts.
    2. Flatten LoRA weights from [max_loras, num_experts, ...] to [max_loras * num_experts, ...].
    3. Run regular SGLang fused-MoE kernels for LoRA A and LoRA B.
    4. Mask out tokens with token_lora_mapping == -1 on the add path.

    ``stage`` splits the call for cross-stream overlap (down-LoRA/finalize overlap):
    - ``"all"`` (default): shrink + expand, returns None — unchanged behavior.
    - ``"shrink"``: routing-A + LoRA-A shrink only (also pre-warms the routing-B cache so the
      expand stage launches no routing kernels). Returns the shrink ``intermediate`` tensor;
      pass a main(consumer)-stream ``intermediate_buffer`` to control its allocation stream.
    - ``"expand"``: routing-B + LoRA-B expand/add only; requires ``intermediate_buffer`` =
      the tensor produced by the ``"shrink"`` stage.

    ``prewarm_a_routing`` and ``prewarm_b_routing`` let a staged caller skip
    routing for a weight that it replaces with a dense operation. The B flag
    also controls the automatic expand-route prewarm performed by ``"shrink"``.
    ``broadcast_intermediate`` is an expand-only mode where one rank vector per
    token is reused for every routed expert.

    ``include_inactive_routes`` retains negative routing blocks. The existing
    expand then writes literal zero to private overwrite outputs, while ADD
    projections preserve their inactive rows. Default routing is unchanged.

    EP accepts either global weights/IDs with a local range or already-localized
    weights/IDs from the standard dispatcher.
    """
    max_loras, _, max_lora_rank, _ = lora_a.shape
    # Global per-expert dim of the LoRA weights. lora_a may be shared-outer (expert
    # dim 1) while lora_b is per-expert, so take the max for the true global count.
    per_expert_dim = max(lora_a.shape[1], lora_b.shape[1])
    ep_local = local_num_experts is not None and local_num_experts < per_expert_dim
    num_routes = topk_ids.numel()
    input_top_k = 1 if hidden_states.shape[0] == num_routes else topk_ids.shape[1]
    if adapter_enabled is not None:
        assert adapter_enabled.shape == (max_loras,)
        assert adapter_enabled.dtype == torch.int32
        assert adapter_enabled.device == topk_ids.device

    def _merge_lora_expert_weight(t: torch.Tensor) -> torch.Tensor:
        # [max_loras, num_experts, x, y] -> [max_loras * num_experts, x, y]
        return t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3])

    def _get_routing(
        topk_ids: torch.Tensor,
        token_lora_mapping: torch.Tensor,
        num_experts: int,
        shared_outer: bool,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check routing_cache for cross-call reuse (gate_up and down share routing)
        cache_key = _routing_cache_key(
            num_experts,
            shared_outer,
            block_size,
            include_inactive_routes,
            adapter_enabled,
        )
        if routing_cache is not None:
            cached = routing_cache.get(cache_key)
            if cached is not None:
                return cached

        # Shared-outer routing has one bucket per adapter, so the same merged
        # align remains valid for multi-LoRA. Compact EP routing stays single-slot.
        compact_merged = ep_local and not shared_outer and max_loras == 1
        bucket_experts = (
            local_num_experts
            if compact_merged
            else (1 if shared_outer else num_experts) * max_loras
        )
        if (
            lora_envs.SGLANG_OPT_LORA_FUSED_MERGED_ALIGN.get()
            and (shared_outer or compact_merged)
            and bucket_experts + 1 <= 1024
            and topk_ids.shape[0] < 512
        ):
            from sglang.kernels.ops.lora.moe.trtllm_lora_temp.moe_lora_merged_align import (
                moe_lora_merged_align,
            )

            (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                token_lora_mask,
                _vne,
            ) = moe_lora_merged_align(
                topk_ids,
                token_lora_mapping,
                num_experts,
                shared_outer,
                max_loras,
                block_size,
                local_expert_offset,
                local_num_experts,
                do_skip=not include_inactive_routes,
                compact=compact_merged,
                adapter_enabled=adapter_enabled,
            )
            result = (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                token_lora_mask,
            )
            if routing_cache is not None:
                routing_cache[cache_key] = result
            return result

        virtual_topk_ids, token_lora_mask, virtual_num_experts = (
            _fused_virtual_topk_ids(
                topk_ids,
                token_lora_mapping,
                num_experts,
                shared_outer,
                max_loras,
                local_expert_offset,
                local_num_experts,
                adapter_enabled=adapter_enabled,
            )
        )
        if ep_local and not shared_outer and local_num_experts < num_experts:
            populated_buckets = local_num_experts * max_loras + 1
        else:
            populated_buckets = virtual_num_experts
        sorted_token_ids, expert_ids, num_tokens_post_padded = _align_virtual_topk_ids(
            virtual_topk_ids, block_size, virtual_num_experts, populated_buckets
        )
        result = (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mask,
        )

        if routing_cache is not None:
            routing_cache[cache_key] = result

        return result

    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )

    assert stage in (
        "all",
        "shrink",
        "expand",
        "routing",
    ), f"invalid stage {stage!r}"
    if broadcast_intermediate:
        assert stage == "expand"
        assert use_direct_expand_add
        assert intermediate_buffer is not None
        assert intermediate_buffer.ndim == 2
        assert intermediate_buffer.shape[0] == token_lora_mapping.shape[0]
    if num_routes == 0:
        if stage == "shrink":
            if intermediate_buffer is not None:
                return intermediate_buffer
            return hidden_states.new_empty(
                (token_lora_mapping.shape[0], topk_ids.shape[1], max_lora_rank)
            )
        if stage == "expand":
            assert intermediate_buffer is not None, (
                "stage='expand' requires intermediate_buffer"
            )
        return None

    num_experts_a = lora_a.shape[1]
    num_experts_b = lora_b.shape[1]
    a_shape = (max_loras * num_experts_a, lora_a.shape[2], lora_a.shape[3])
    b_shape = (lora_b.shape[0] * num_experts_b, lora_b.shape[2], lora_b.shape[3])
    tokens = token_lora_mapping.shape[0]
    needs_shrink_config = stage in ("all", "shrink") or (
        stage == "routing" and prewarm_a_routing
    )
    reuse_b_block = (
        needs_shrink_config
        and tokens >= 512
        and lora_envs.SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE.get()
    )
    needs_b_config = (
        stage in ("all", "expand")
        or (prewarm_b_routing and (stage == "routing" or routing_cache is not None))
        or reuse_b_block
    )
    b_stage_config = (
        _get_stage_config(b_shape, 1, hidden_states.dtype, tokens)
        if needs_b_config
        else None
    )

    if stage == "routing":
        # Pre-warm the routing cache on the CALLER'S (main) stream so the
        # side-stream chain performs no allocations. Tensors allocated inside a
        # side-stream context during cuda-graph capture can be pool-reused by
        # later allocations on other streams with no cross-stream guard (the
        # allocator's stream tracking is disabled while capturing), corrupting
        # replays. Routing needs only topk_ids + token_lora_mapping, which are
        # both ready before the side-stream fork, so it can run on main.
        if prewarm_a_routing:
            a_cfg = _get_shrink_stage_config(
                a_shape,
                tokens,
                b_stage_config["BLOCK_SIZE_M"] if reuse_b_block else None,
            )
            _get_routing(
                topk_ids,
                token_lora_mapping,
                num_experts_a,
                experts_shared_outer_loras_a,
                a_cfg["BLOCK_SIZE_M"],
            )
        if prewarm_b_routing:
            assert b_stage_config is not None
            _get_routing(
                topk_ids,
                token_lora_mapping,
                num_experts_b,
                experts_shared_outer_loras_b,
                b_stage_config["BLOCK_SIZE_M"],
            )
        return None

    intermediate = intermediate_buffer
    if stage != "expand":
        a_stage_config = _get_shrink_stage_config(
            a_shape, tokens, b_stage_config["BLOCK_SIZE_M"] if reuse_b_block else None
        )
        lora_a_virtual = _merge_lora_expert_weight(lora_a)
        (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mask,
        ) = _get_routing(
            topk_ids,
            token_lora_mapping,
            num_experts_a,
            experts_shared_outer_loras_a,
            a_stage_config["BLOCK_SIZE_M"],
        )
        intermediate_shape = [
            token_lora_mapping.shape[0],
            topk_ids.shape[1],
            max_lora_rank,
        ]
        intermediate_split_k = _get_moe_lora_shrink_split_k(
            lora_a_virtual, sorted_token_ids, a_stage_config
        )
        # EP leaves non-owned [token, k] shrink slots unwritten. A per-expert expand skips
        # non-owned blocks (never reads them), but a shared-outer expand routes by lora id
        # and would read them into the real (all-reduced) output -> must zero. split_k > 1
        # also needs a zeroed buffer for its accumulation.
        must_zero_intermediate = (
            zero_intermediate
            or intermediate_split_k > 1
            or (ep_local and experts_shared_outer_loras_b)
        )
        if intermediate is None:
            intermediate = (
                torch.zeros(
                    intermediate_shape,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                if must_zero_intermediate
                else torch.empty(
                    intermediate_shape,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            )
        elif must_zero_intermediate:
            # Caller-provided buffer (allocated on the consumer stream): zero it in-stream.
            intermediate.zero_()

        _invoke_moe_lora_shrink_splitk(
            hidden_states,
            lora_a_virtual,
            intermediate.view(-1, max_lora_rank),
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            input_top_k,
            a_stage_config,
        )

        if stage == "shrink":
            # Pre-warm the routing-B cache on this (side) stream so the later "expand" stage
            # launches no routing kernels — they overlap finalize together with the shrink.
            if routing_cache is not None and prewarm_b_routing:
                assert b_stage_config is not None
                _get_routing(
                    topk_ids,
                    token_lora_mapping,
                    num_experts_b,
                    experts_shared_outer_loras_b,
                    b_stage_config["BLOCK_SIZE_M"],
                )
            return intermediate

    assert intermediate is not None, "stage='expand' requires intermediate_buffer"
    assert b_stage_config is not None
    lora_b_virtual = _merge_lora_expert_weight(lora_b)
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_b,
        experts_shared_outer_loras_b,
        b_stage_config["BLOCK_SIZE_M"],
    )

    intermediate_flat = intermediate.view(-1, max_lora_rank)
    # Shared-add overlap: the expand below adds into the SAME output buffer the
    # overlapped shared-expert add (enqueued on the main stream by the LoRA
    # dispatch) is writing — wait for it here, right before the expand launch,
    # so the shrink + stage-B routing above still overlap the add.
    if expand_wait_event is not None:
        torch.cuda.current_stream().wait_event(expand_wait_event)
    # The rank-specialized direct expand-add doesn't support the shared-outer
    # LoRA-B layout; fall back to the generic kernel for that case instead of
    # asserting, so adapters with experts_shared_outer_loras still work.
    if use_direct_expand_add and not experts_shared_outer_loras_b:
        assert not fuse_add_to_output
        _invoke_moe_lora_expand_add(
            intermediate_flat,
            lora_b_virtual,
            output,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            b_stage_config,
            mul_routed_weight,
            fuse_sum_all_reduce,
            broadcast_intermediate=broadcast_intermediate,
        )
    else:
        assert not broadcast_intermediate, (
            "broadcasted LoRA-A intermediates require the rank-specialized "
            "direct expand kernel"
        )
        invoke_fused_moe_kernel(
            intermediate_flat,
            lora_b_virtual,
            None,
            output,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            1,
            b_stage_config,
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16,
            False,
            False,
            False,
            False,
            False,
            None,
            fuse_add_to_output=fuse_add_to_output,
            fuse_sum_all_reduce=fuse_sum_all_reduce,
            lora_preserve_base=True,
            add_output_mask=token_lora_mask,
            mask_output=not fuse_add_to_output and not fuse_sum_all_reduce,
            router_topk=topk_ids.shape[1],
        )


def _merged_experts_fused_moe_lora_add_op(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
) -> None:
    _merged_experts_fused_moe_lora_add_impl(
        output,
        hidden_states,
        lora_a,
        lora_b,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        mul_routed_weight,
        experts_shared_outer_loras_a,
        experts_shared_outer_loras_b,
    )


from sglang.srt.utils.common import direct_register_custom_op

direct_register_custom_op(
    op_name="merged_experts_fused_moe_lora_add",
    op_func=_merged_experts_fused_moe_lora_add_op,
    mutates_args=["output"],
    fake_impl=_merged_experts_fused_moe_lora_add_fake,
)


def merged_experts_fused_moe_lora_add(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    routing_cache: dict | None = None,
    fuse_add_to_output: bool = True,
    fuse_sum_all_reduce: bool = False,
    use_direct_expand_add: bool = False,
    local_expert_offset: int = 0,
    local_num_experts: int | None = None,
    stage: str = "all",
    intermediate_buffer: torch.Tensor | None = None,
    expand_wait_event: "torch.cuda.Event | None" = None,
    broadcast_intermediate: bool = False,
    prewarm_a_routing: bool = True,
    prewarm_b_routing: bool = True,
    zero_intermediate: bool = False,
    include_inactive_routes: bool = False,
    adapter_enabled: torch.Tensor | None = None,
) -> "torch.Tensor | None":
    """Public API: wraps the registered op with routing_cache support."""
    return _merged_experts_fused_moe_lora_add_impl(
        output,
        hidden_states,
        lora_a,
        lora_b,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        mul_routed_weight,
        experts_shared_outer_loras_a,
        experts_shared_outer_loras_b,
        routing_cache,
        fuse_add_to_output=fuse_add_to_output,
        fuse_sum_all_reduce=fuse_sum_all_reduce,
        use_direct_expand_add=use_direct_expand_add,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        stage=stage,
        intermediate_buffer=intermediate_buffer,
        expand_wait_event=expand_wait_event,
        broadcast_intermediate=broadcast_intermediate,
        prewarm_a_routing=prewarm_a_routing,
        prewarm_b_routing=prewarm_b_routing,
        zero_intermediate=zero_intermediate,
        include_inactive_routes=include_inactive_routes,
        adapter_enabled=adapter_enabled,
    )
