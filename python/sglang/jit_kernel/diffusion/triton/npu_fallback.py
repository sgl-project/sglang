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


def fused_rmsnorm_rope_native(x, weight, cos, sin, head_dim, eps):
    orig_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps) * weight.float()
    shape = x_normed.shape
    x_normed = x_normed.view(*shape[:-1], -1, head_dim)
    x1 = x_normed[..., ::2]
    x2 = x_normed[..., 1::2]
    cos_v = cos.unsqueeze(-2).to(x_normed.dtype)
    sin_v = sin.unsqueeze(-2).to(x_normed.dtype)
    o1 = x1 * cos_v - x2 * sin_v
    o2 = x1 * sin_v + x2 * cos_v
    out = torch.stack((o1, o2), dim=-1).flatten(-2).flatten(-2)
    return out.to(orig_dtype)


def fused_rmsnorm_rope_tp_native(
    x, norm_module, cos, sin, head_dim, tp_rank, tp_size, tp_group
):
    import torch._C._distributed_c10d as c10d

    weight_local = norm_module.weight.tensor_split(tp_size)[tp_rank].float()
    eps = norm_module.variance_epsilon
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    variance = tp_group.all_reduce(variance, op=c10d.ReduceOp.AVG)
    x_normed = x_fp32 * torch.rsqrt(variance + eps) * weight_local
    shape = x_normed.shape
    x_normed = x_normed.view(*shape[:-1], -1, head_dim)
    x1 = x_normed[..., ::2]
    x2 = x_normed[..., 1::2]
    cos_v = cos.unsqueeze(-2).to(x_normed.dtype)
    sin_v = sin.unsqueeze(-2).to(x_normed.dtype)
    o1 = x1 * cos_v - x2 * sin_v
    o2 = x1 * sin_v + x2 * cos_v
    out = torch.stack((o1, o2), dim=-1).flatten(-2).flatten(-2)
    return out.to(x.dtype)
