"""Apply GGUF route weights without reading inactive expert outputs."""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["weight_stride0", "weight_stride1", "id_stride0", "id_stride1"]
)
def _apply_gguf_moe_weights_kernel(
    outputs,
    weights,
    expert_ids,
    width: tl.constexpr,
    topk: tl.constexpr,
    weight_stride0,
    weight_stride1,
    id_stride0,
    id_stride1,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = row // topk
    slot = row % topk
    active = tl.load(expert_ids + token * id_stride0 + slot * id_stride1) >= 0
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        outputs + row * width + columns,
        mask=active & (columns < width),
        other=0,
    ).to(tl.float32)
    factor = tl.load(
        weights + token * weight_stride0 + slot * weight_stride1,
        mask=active,
        other=0,
    ).to(tl.float32)
    tl.store(outputs + row * width + columns, values * factor, mask=columns < width)


def apply_gguf_moe_weights(
    outputs: torch.Tensor, weights: torch.Tensor, expert_ids: torch.Tensor
) -> None:
    """Weight contiguous GGUF rows in place, zeroing routes with negative IDs.

    ``outputs`` is ``[tokens * topk, hidden]`` and routing tensors are
    ``[tokens, topk]``. Products round to the output dtype before the caller's
    native ``moe_sum``; inactive rows read neither outputs nor weights.
    """
    assert outputs.ndim == weights.ndim == expert_ids.ndim == 2
    assert outputs.is_contiguous()
    assert weights.shape == expert_ids.shape
    assert outputs.shape[0] == expert_ids.numel()
    assert outputs.device == weights.device == expert_ids.device
    assert outputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert expert_ids.dtype in (torch.int32, torch.int64)
    if outputs.numel() == 0:
        return
    width = outputs.shape[1]
    block = min(1024, triton.next_power_of_2(width))
    _apply_gguf_moe_weights_kernel[(outputs.shape[0], triton.cdiv(width, block))](
        outputs,
        weights,
        expert_ids,
        width,
        weights.shape[1],
        weights.stride(0),
        weights.stride(1),
        expert_ids.stride(0),
        expert_ids.stride(1),
        BLOCK=block,
    )
