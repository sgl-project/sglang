import torch
import torch_npu

NPU_ROTARY_MUL_MAX_NUM_HEADS = 1000
NPU_ROTARY_MUL_MAX_HEAD_SIZE = 896


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

    if (
        cos.dim() == 3
        and x.dim() == 3
        and x.shape[1] < NPU_ROTARY_MUL_MAX_NUM_HEADS
        and x.shape[2] < NPU_ROTARY_MUL_MAX_HEAD_SIZE
    ):
        if cos.size(-1) * 2 == x.size(-1):
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        x = x.unsqueeze(0)
        x_embed = torch_npu.npu_rotary_mul(x, cos, sin)
        x_embed = x_embed.squeeze(0)
        return x_embed

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)
