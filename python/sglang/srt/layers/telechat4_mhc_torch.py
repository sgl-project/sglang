import torch
import torch.nn.functional as F


def telechat4_mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_streams = residual.shape[-2]
    residual_fp32 = residual.float()
    residual_flat = residual_fp32.flatten(-2)
    inv_rms = torch.rsqrt(residual_flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(residual_flat, fn.float()) * inv_rms
    pre_logits, post_logits, comb_logits = torch.split(
        mixes,
        [num_streams, num_streams, num_streams * num_streams],
        dim=-1,
    )

    pre_mix = (
        torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:num_streams]) + hc_pre_eps
    )
    post_mix = (
        torch.sigmoid(
            post_logits * hc_scale[1] + hc_base[num_streams : 2 * num_streams]
        )
        * hc_post_mult_value
    )
    comb_mix = (comb_logits * hc_scale[2] + hc_base[2 * num_streams :]).view(
        -1, num_streams, num_streams
    )
    comb_mix = torch.exp(comb_mix - comb_mix.amax(dim=-1, keepdim=True))
    comb_mix = comb_mix / comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (pre_mix.unsqueeze(-1) * residual_fp32).sum(dim=-2)
    return post_mix.unsqueeze(-1), comb_mix, layer_input.to(torch.bfloat16)


def telechat4_mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    output = post_mix.squeeze(-1).unsqueeze(-1) * x.float().unsqueeze(-2)
    output += (comb_mix.unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(dim=-3)
    return output.to(torch.bfloat16)
