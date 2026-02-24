import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_device_sm, is_hip

_AUTOTUNE_BLOCK_V_OPTIONS = (1024, 2048, 4096, 8192)
_AUTOTUNE_NUM_WARPS_OPTIONS = (4, 8)
_FALLBACK_BLOCK_V_SMALL_THRESHOLD = 2048
_FALLBACK_BLOCK_V_MEDIUM_THRESHOLD = 8192
_FALLBACK_BLOCK_V_MEDIUM = 2048
_FALLBACK_BLOCK_V_LARGE = 4096

_IS_HIP = is_hip()
_DEVICE_SM = get_device_sm()


def _get_num_stages_options():
    if _IS_HIP:
        return (2,)
    if _DEVICE_SM >= 120:
        return (2, 3, 4)
    if _DEVICE_SM >= 90:
        return (2, 4)
    return (2, 3)


def _get_fallback_num_warps():
    if _IS_HIP:
        return 4
    if _DEVICE_SM >= 90:
        return 8
    return 4


_AUTOTUNE_NUM_STAGES_OPTIONS = _get_num_stages_options()
_FALLBACK_NUM_WARPS = _get_fallback_num_warps()


def _select_fallback_block_v(vocab_size: int) -> int:
    if vocab_size <= _FALLBACK_BLOCK_V_SMALL_THRESHOLD:
        return triton.next_power_of_2(vocab_size)
    if vocab_size <= _FALLBACK_BLOCK_V_MEDIUM_THRESHOLD:
        return _FALLBACK_BLOCK_V_MEDIUM
    return _FALLBACK_BLOCK_V_LARGE


@triton.jit
def _dllm_post_process_kernel(
    logits_ptr,
    input_ids_ptr,
    transfer_out_ptr,
    confidence_out_ptr,
    argmax_out_ptr,
    num_transfers_ptr,
    mask_id: tl.constexpr,
    threshold: tl.constexpr,
    block_size,
    vocab_size,
    logits_stride,
    BLOCK_V: tl.constexpr,
):
    row_idx = tl.program_id(0)

    if row_idx >= block_size:
        return

    row_start = row_idx * logits_stride

    input_id = tl.load(input_ids_ptr + row_idx)
    is_masked = input_id == mask_id

    if not is_masked:
        tl.store(transfer_out_ptr + row_idx, 0)
        tl.store(confidence_out_ptr + row_idx, -float("inf"))
        tl.store(argmax_out_ptr + row_idx, input_id)
        return

    max_val = -float("inf")
    exp_sum = 0.0
    argmax_idx = 0

    for v_start in range(0, vocab_size, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offsets < vocab_size

        logits = tl.load(
            logits_ptr + row_start + v_offsets, mask=v_mask, other=-float("inf")
        )
        logits = logits.to(tl.float32)

        tile_max = tl.max(logits)

        if tile_max > max_val:
            exp_sum = exp_sum * tl.exp(max_val - tile_max)
            max_val = tile_max

            is_max = logits == tile_max
            tile_indices = tl.where(is_max, v_offsets, vocab_size)
            argmax_idx = tl.min(tile_indices)

        exp_vals = tl.exp(logits - max_val)
        exp_sum += tl.sum(tl.where(v_mask, exp_vals, 0.0))

    prob = 1.0 / exp_sum
    transfer = prob > threshold

    tl.store(argmax_out_ptr + row_idx, argmax_idx)
    tl.store(confidence_out_ptr + row_idx, prob)
    tl.store(transfer_out_ptr + row_idx, transfer.to(tl.int8))

    if transfer:
        tl.store(input_ids_ptr + row_idx, argmax_idx)
        tl.atomic_add(num_transfers_ptr, 1)


@triton.jit
def _dllm_fallback_kernel(
    input_ids_ptr,
    confidence_ptr,
    argmax_out_ptr,
    transfer_out_ptr,
    num_transfers_ptr,
    block_size,
):
    num_transfers = tl.load(num_transfers_ptr)
    if num_transfers > 0:
        return

    best_idx = 0
    best_conf = -float("inf")
    for i in range(block_size):
        c = tl.load(confidence_ptr + i)
        if c > best_conf:
            best_conf = c
            best_idx = i

    argmax_token = tl.load(argmax_out_ptr + best_idx)
    tl.store(input_ids_ptr + best_idx, argmax_token)
    tl.store(transfer_out_ptr + best_idx, tl.cast(1, tl.int8))
    tl.store(num_transfers_ptr, 1)


_dllm_post_process_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": bv}, num_warps=nw, num_stages=ns)
        for bv in _AUTOTUNE_BLOCK_V_OPTIONS
        for nw in _AUTOTUNE_NUM_WARPS_OPTIONS
        for ns in _AUTOTUNE_NUM_STAGES_OPTIONS
    ],
    key=["vocab_size"],
)(_dllm_post_process_kernel)


def calculate_low_confidence_score(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask_id: int,
    threshold: float,
    autotune: bool = True,
) -> None:
    block_size, vocab_size = logits.shape
    device = logits.device

    transfer_index = torch.empty(block_size, dtype=torch.int8, device=device)
    confidence = torch.empty(block_size, dtype=torch.float32, device=device)
    argmax_tokens = torch.empty(block_size, dtype=torch.int64, device=device)
    num_transfers_dev = torch.zeros(1, dtype=torch.int32, device=device)

    grid = (block_size,)

    if autotune:
        _dllm_post_process_kernel_autotuned[grid](
            logits,
            input_ids,
            transfer_index,
            confidence,
            argmax_tokens,
            num_transfers_dev,
            mask_id,
            threshold,
            block_size,
            vocab_size,
            logits.stride(0),
        )
    else:
        BLOCK_V = _select_fallback_block_v(vocab_size)

        _dllm_post_process_kernel[grid](
            logits,
            input_ids,
            transfer_index,
            confidence,
            argmax_tokens,
            num_transfers_dev,
            mask_id,
            threshold,
            block_size,
            vocab_size,
            logits.stride(0),
            BLOCK_V=BLOCK_V,
            num_warps=_FALLBACK_NUM_WARPS,
        )

    _dllm_fallback_kernel[(1,)](
        input_ids,
        confidence,
        argmax_tokens,
        transfer_index,
        num_transfers_dev,
        block_size,
    )
