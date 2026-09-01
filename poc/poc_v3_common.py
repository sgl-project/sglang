"""Shared helpers for v3 SoA fp8 unified_kv decode PoC."""
import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode_soa_v3 import (
    _FP8_DTYPE, _DIM_NOPE, _DIM_ROPE, _GROUP, _NUM_G, _NUM_G_PAD,
)


def quant_soa(kv_bf16):
    """[C,512] bf16 -> (nope_fp8[C,448], rope_bf16[C,64], scale_f32[C,8]).
    RoPE kept exactly bf16; nope 1x64 block-scale fp8. Returns also the
    bf16 ground-truth dequant for oracle."""
    C, D = kv_bf16.shape
    assert D == _DIM_NOPE + _DIM_ROPE
    dev = kv_bf16.device
    nope = kv_bf16[:, :_DIM_NOPE].float().view(C, _NUM_G, _GROUP)
    fmax = torch.finfo(_FP8_DTYPE).max
    amax = nope.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = (amax / fmax)  # [C,7,1]
    q = (nope / scale).clamp(-fmax, fmax).to(_FP8_DTYPE)  # [C,7,64]
    nope_fp8 = q.reshape(C, _DIM_NOPE).contiguous()
    rope_bf16 = kv_bf16[:, _DIM_NOPE:].contiguous()
    scale_f32 = torch.ones((C, _NUM_G_PAD), dtype=torch.float32, device=dev)
    scale_f32[:, :_NUM_G] = scale.squeeze(-1)
    scale_f32 = scale_f32.contiguous()

    # ground-truth dequant -> bf16
    nope_dq = (q.float() * scale).reshape(C, _DIM_NOPE)
    unified_dq = torch.cat([nope_dq.to(torch.bfloat16), rope_bf16], dim=1).contiguous()
    return nope_fp8, rope_bf16, scale_f32, unified_dq
