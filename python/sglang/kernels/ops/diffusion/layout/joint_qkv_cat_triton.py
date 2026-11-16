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
    if not (
        img_q.is_cuda
        and img_q.dtype in (torch.float16, torch.bfloat16)
        and img_q.ndim == 4
        and min(img_q.shape) > 0
        and img_q.shape[-2] * img_q.shape[-1] <= 8192
    ):
        raise RuntimeError(
            "joint QKV expects FP16/BF16 CUDA [B, S, H, D], 0 < H * D <= 8192"
        )
    for index, value in enumerate(inputs):
        expected = img_q.shape if index < 3 else txt_q.shape
        if not (
            value.ndim == 4
            and value.shape == expected
            and value.shape[0] == img_q.shape[0]
            and value.shape[2:] == img_q.shape[2:]
            and value.shape[1] > 0
            and value.dtype == img_q.dtype
            and value.device == img_q.device
            and not value.requires_grad
            and value.stride(-1) == 1
            and value.stride(-2) == img_q.shape[-1]
        ):
            raise RuntimeError(
                "joint QKV inputs must share batch/head shape, dtype/device and have packed head rows without gradients"
            )
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
