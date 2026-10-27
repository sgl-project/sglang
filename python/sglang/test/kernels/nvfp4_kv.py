"""NVFP4 KV quantize in `NVFP4KVQuantizeUtil`'s layout, in torch: the gfx950 M3.1 kernel
tests build their packed caches with it (the pool stores through the fused qk-norm kernel)."""

import torch

# e2m1 magnitude code = how many of these the value passes: round half to even, saturating at 6
_E2M1_ABOVE = (0.25, 1.25, 2.5, 5.0)
_E2M1_AT_OR_ABOVE = (0.75, 1.75, 3.5)


def nvfp4_kv_quantize(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[R, D]`` bf16/fp16 -> (``[R, D//2]`` uint8 payload, ``[R, D//16]`` uint8 E4M3 scales)."""
    rows, dim = x.shape
    assert dim % 16 == 0
    gs = global_scale.float().reshape(())
    blocks = x.float().view(rows, dim // 16, 16)
    amax = blocks.abs().amax(-1)
    # saturate at E4M3's 448 like the hardware conversion; torch's cast overflows to NaN
    scale_e4m3 = (amax / (6.0 * gs)).clamp(max=448.0).to(torch.float8_e4m3fn)
    inv = torch.where(amax != 0, 1.0 / (scale_e4m3.float() * gs), 0.0)
    v = blocks * inv[..., None]
    a = v.abs()
    mag = sum((a > t).int() for t in _E2M1_ABOVE) + sum(
        (a >= t).int() for t in _E2M1_AT_OR_ABOVE
    )
    codes = (mag | torch.where(v < 0, 8, 0)).to(torch.uint8).view(rows, dim // 2, 2)
    payload = codes[..., 0] | (codes[..., 1] << 4)
    return payload, scale_e4m3.view(torch.uint8)
