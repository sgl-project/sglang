import torch


# TODO: remove this when triton ascend bug is fixed
def fuse_scale_shift_native(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    block_l: int = 128,
    block_c: int = 128,
):
    return x * (1 + scale) + shift


# TODO: remove this when triton ascend bug is fixed
def apply_rotary_embedding_native(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)
