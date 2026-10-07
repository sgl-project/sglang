"""Copy image/text QKV views into contiguous joint attention inputs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _joint_qkv_cat_kernel(
    IQ,
    IK,
    IV,
    TQ,
    TK,
    TV,
    OUT,
    IMAGE_TOKENS: tl.constexpr,
    TEXT_TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BATCH: tl.constexpr,
    IMAGE_STRIDES: tl.constexpr,
    TEXT_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    component = tl.program_id(1)
    sequence = IMAGE_TOKENS + TEXT_TOKENS
    batch = row // sequence
    token = row % sequence
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    # Each program copies one Q, K or V row. Separate strides preserve packed
    # V views when Q/K have already been normalized into contiguous tensors.
    if component == 0:
        image, text = IQ, TQ
        ib, it = IMAGE_STRIDES[0], IMAGE_STRIDES[1]
        tb, tt = TEXT_STRIDES[0], TEXT_STRIDES[1]
    elif component == 1:
        image, text = IK, TK
        ib, it = IMAGE_STRIDES[2], IMAGE_STRIDES[3]
        tb, tt = TEXT_STRIDES[2], TEXT_STRIDES[3]
    else:
        image, text = IV, TV
        ib, it = IMAGE_STRIDES[4], IMAGE_STRIDES[5]
        tb, tt = TEXT_STRIDES[4], TEXT_STRIDES[5]
    if token < IMAGE_TOKENS:
        value = tl.load(image + batch * ib + token * it + cols, mask, other=0)
    else:
        value = tl.load(
            text + batch * tb + (token - IMAGE_TOKENS) * tt + cols,
            mask,
            other=0,
        )
    # No floating-point arithmetic: preserve signed zeros and NaN payloads.
    tl.store(OUT + (component * BATCH * sequence + row) * HIDDEN + cols, value, mask)


def can_use_joint_qkv_cat(*inputs: torch.Tensor) -> bool:
    if len(inputs) != 6 or torch.compiler.is_compiling() or torch.version.hip:
        return False
    first = inputs[0]
    if first.ndim != 4 or not first.is_cuda:
        return False
    if first.dtype not in (torch.float16, torch.bfloat16):
        return False
    batch, tokens, heads, dim = first.shape
    if min(batch, tokens, heads, dim) <= 0 or heads * dim > 8192:
        return False
    for index, value in enumerate(inputs):
        expected = first.shape if index < 3 else inputs[3].shape
        if (
            value.ndim != 4
            or value.shape != expected
            or value.shape[0] != batch
            or value.shape[2:] != (heads, dim)
            or value.shape[1] <= 0
            or value.dtype != first.dtype
            or value.device != first.device
            or value.requires_grad
            or value.stride(-1) != 1
            or value.stride(-2) != dim
        ):
            return False
    return True


def joint_qkv_cat(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate three pairs of ``[B, S, H, D]`` tensors, image first.

    Batch/token strides may differ across inputs; each head row is contiguous.
    The three outputs occupy disjoint, contiguous regions of one allocation.
    """
    inputs = (img_q, img_k, img_v, txt_q, txt_k, txt_v)
    assert can_use_joint_qkv_cat(*inputs)
    batch, image_tokens, heads, dim = img_q.shape
    text_tokens = txt_q.shape[1]
    output = torch.empty(
        (3, batch, image_tokens + text_tokens, heads, dim),
        device=img_q.device,
        dtype=img_q.dtype,
    )
    image_strides = tuple(s for x in inputs[:3] for s in x.stride()[:2])
    text_strides = tuple(s for x in inputs[3:] for s in x.stride()[:2])
    with torch.cuda.device(img_q.device):
        _joint_qkv_cat_kernel[(batch * (image_tokens + text_tokens), 3)](
            *inputs,
            output,
            image_tokens,
            text_tokens,
            heads * dim,
            batch,
            image_strides,
            text_strides,
            BLOCK=triton.next_power_of_2(heads * dim),
            num_warps=4,
        )
    return output.unbind(0)
