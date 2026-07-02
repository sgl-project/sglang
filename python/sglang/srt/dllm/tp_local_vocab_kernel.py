import torch
import triton
import triton.language as tl

from sglang.srt.dllm.tp_local_vocab_state import VocabState


LOCAL_VOCAB_STATE_TRITON_MAX_BLOCK_VOCAB = 131072


@triton.jit
def _local_vocab_state_kernel(
    logits_ptr,
    penalty_token_ids_ptr,
    max_values_ptr,
    argmax_ids_ptr,
    logsumexp_ptr,
    stride_row: tl.constexpr,
    valid_vocab_size: tl.constexpr,
    vocab_start: tl.constexpr,
    penalty_lambda: tl.constexpr,
    APPLY_PENALTY: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_VOCAB)
    valid = offsets < valid_vocab_size
    values = tl.load(
        logits_ptr + row_id * stride_row + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    if APPLY_PENALTY:
        penalty_token_id = tl.load(penalty_token_ids_ptr + row_id)
        penalty_offset = penalty_token_id - vocab_start
        values = values - tl.where(
            valid & (offsets == penalty_offset),
            penalty_lambda,
            0.0,
        )

    max_value = tl.max(values, axis=0)
    arg_offsets = tl.min(tl.where(values == max_value, offsets, BLOCK_VOCAB), axis=0)
    exp_sum = tl.sum(tl.exp(values - max_value), axis=0)
    logsumexp = max_value + tl.log(exp_sum)

    tl.store(max_values_ptr + row_id, max_value)
    tl.store(argmax_ids_ptr + row_id, arg_offsets + vocab_start)
    tl.store(logsumexp_ptr + row_id, logsumexp)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def can_use_local_vocab_state_triton(valid_vocab_size: int) -> bool:
    return (
        _next_power_of_2(valid_vocab_size)
        <= LOCAL_VOCAB_STATE_TRITON_MAX_BLOCK_VOCAB
    )


def local_vocab_state_from_logits_triton(
    *,
    local_logits: torch.Tensor,
    vocab_start: int,
    valid_vocab_size: int | None = None,
    penalized_token_ids: torch.Tensor | None = None,
    penalty_lambda: float = 0.0,
) -> VocabState:
    if not local_logits.is_cuda:
        raise ValueError("local_vocab_state_from_logits_triton requires a CUDA tensor")
    if local_logits.dim() != 2:
        raise ValueError("local_logits must be a 2D tensor")

    valid_vocab_size = (
        local_logits.shape[-1] if valid_vocab_size is None else int(valid_vocab_size)
    )
    if valid_vocab_size <= 0 or valid_vocab_size > local_logits.shape[-1]:
        raise ValueError("valid_vocab_size must be in [1, local_logits.shape[-1]]")

    local_logits = local_logits.contiguous()
    num_rows = local_logits.shape[0]
    max_values = torch.empty((num_rows,), device=local_logits.device, dtype=torch.float32)
    argmax_ids = torch.empty((num_rows,), device=local_logits.device, dtype=torch.long)
    logsumexp = torch.empty((num_rows,), device=local_logits.device, dtype=torch.float32)

    block_vocab = _next_power_of_2(valid_vocab_size)
    apply_penalty = penalized_token_ids is not None and penalty_lambda > 0
    if apply_penalty:
        penalty_token_ids = penalized_token_ids.to(device=local_logits.device).contiguous()
    else:
        penalty_token_ids = torch.empty((num_rows,), device=local_logits.device, dtype=torch.long)

    _local_vocab_state_kernel[(num_rows,)](
        local_logits,
        penalty_token_ids,
        max_values,
        argmax_ids,
        logsumexp,
        local_logits.stride(0),
        valid_vocab_size,
        int(vocab_start),
        float(penalty_lambda),
        APPLY_PENALTY=apply_penalty,
        BLOCK_VOCAB=block_vocab,
        num_warps=8,
    )
    return VocabState(
        max_values=max_values,
        argmax_ids=argmax_ids,
        logsumexp=logsumexp,
        max_probs=torch.exp(max_values - logsumexp),
    )
