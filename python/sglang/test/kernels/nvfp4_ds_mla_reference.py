"""Pure-torch reference for the nvfp4_ds_mla KV writer
(``set_mla_kv_buffer_nvfp4_ds_mla_triton``): the 352 B NVFP4 DSA MLA record."""

import torch

from sglang.kernels.ops.kvcache.mla_buffer import (
    _NVFP4_MIN_BLOCK_SCALE,
    NVFP4_DS_MLA_ROW_BYTES,
)


def _e2m1_codes_ref(v: torch.Tensor) -> torch.Tensor:
    a = v.abs()
    code = (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )
    return torch.where(v < 0, code | 8, code)


def quantize_nvfp4_ds_mla_reference(
    k_nope: torch.Tensor, k_rope: torch.Tensor
) -> torch.Tensor:
    """Torch reference of ``set_mla_kv_buffer_nvfp4_ds_mla_kernel``: [T, 352] uint8."""
    x = k_nope.reshape(-1, 32, 16).float()
    t = x.shape[0]
    amax = x.abs().amax(-1)
    scale_e4m3 = (amax / 6.0).clamp_min(_NVFP4_MIN_BLOCK_SCALE).to(torch.float8_e4m3fn)
    v = x / scale_e4m3.float()[..., None]
    code = _e2m1_codes_ref(v).reshape(t, 256, 2)
    out = torch.empty(t, NVFP4_DS_MLA_ROW_BYTES, dtype=torch.uint8, device=x.device)
    out[:, :256] = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    out[:, 256:320] = (
        k_rope.reshape(t, 64).float().to(torch.float8_e4m3fn).view(torch.uint8)
    )
    out[:, 320:] = (
        scale_e4m3.view(torch.uint8).unflatten(-1, (8, 4)).transpose(-1, -2).flatten(-2)
    )
    return out
