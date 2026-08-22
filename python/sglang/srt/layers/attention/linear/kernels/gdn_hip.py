from __future__ import annotations

from typing import Any

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.utils import is_gfx95_supported, is_hip

if is_hip() and is_gfx95_supported():
    from aiter import gdr_decode_packed_bf16
else:
    gdr_decode_packed_bf16 = None


class HipGDNKernel(TritonGDNKernel):
    """gfx950 packed BF16 GDN decode with Triton for all other paths."""

    supports_packed_decode: bool = is_hip() and is_gfx95_supported()

    @staticmethod
    def _uses_replayssm(kwargs: dict[str, Any]) -> bool:
        return all(
            kwargs.get(name) is not None
            for name in (
                "replayssm_d",
                "replayssm_k",
                "replayssm_g",
                "replayssm_write_pos",
            )
        )

    @staticmethod
    def _matches_exact_contract(
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
    ) -> bool:
        batch = mixed_qkv.shape[0] if mixed_qkv.ndim == 2 else -1
        tensors = (mixed_qkv, a, b, A_log, dt_bias, ssm_states, cache_indices)
        return (
            is_hip()
            and is_gfx95_supported()
            and mixed_qkv.ndim == 2
            and mixed_qkv.shape[1] == 6144
            and mixed_qkv.dtype == torch.bfloat16
            and mixed_qkv.stride(1) == 1
            and mixed_qkv.stride(0) * mixed_qkv.element_size() % 16 == 0
            and mixed_qkv.data_ptr() % 16 == 0
            and a.shape == (batch, 32)
            and b.shape == (batch, 32)
            and a.dtype == torch.bfloat16
            and b.dtype == torch.bfloat16
            and a.stride(1) == 1
            and b.stride(1) == 1
            and dt_bias.shape == (32,)
            and dt_bias.dtype == torch.bfloat16
            and dt_bias.stride(0) == 1
            and A_log.shape == (32,)
            and A_log.dtype == torch.float32
            and A_log.stride(0) == 1
            and cache_indices.shape == (batch,)
            and cache_indices.dtype == torch.int32
            and cache_indices.stride(0) >= 1
            and ssm_states.ndim == 4
            and ssm_states.shape[1:] == (32, 128, 128)
            and ssm_states.dtype == torch.bfloat16
            and ssm_states.stride()[1:] == (128 * 128, 128, 1)
            and ssm_states.data_ptr() % 16 == 0
            and ssm_states.stride(0) * ssm_states.element_size() % 16 == 0
            and num_v_heads == 32
            and head_v_dim == 128
            and abs(float(scale) - 128**-0.5) <= 1e-12
            and all(t.device == mixed_qkv.device for t in tensors)
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        parent_kwargs = dict(
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )
        if self._uses_replayssm(kwargs) or not self._matches_exact_contract(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
        ):
            return super().packed_decode(mixed_qkv, a, b, **parent_kwargs)

        if gdr_decode_packed_bf16 is None:
            raise RuntimeError("GDN HIP backend is unavailable outside ROCm gfx95")

        batch = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(batch, 1, num_v_heads, head_v_dim)
        gdr_decode_packed_bf16(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            dt_bias=dt_bias,
            A_log=A_log,
            indices=cache_indices,
            state=ssm_states,
            out=out,
            scale=scale,
        )
        return out.transpose(0, 1)
