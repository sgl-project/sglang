"""Helion backend for Kimi Delta Attention."""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


class HelionKDAKernel(LinearAttnKernelBase):
    """KDA packed decode and prefill implemented with Helion kernels.

    The generic decode interface delegates to Triton, and the dispatcher routes
    speculative target verification directly to Triton. The one-token decode
    and ReplaySSM paths use :meth:`packed_decode`, while prefill uses
    :meth:`extend`.
    """

    supports_packed_decode = True

    def __init__(
        self,
        triton_fallback: TritonKDAKernel | None = None,
        *,
        enable_decode: bool = True,
        enable_prefill: bool = True,
    ) -> None:
        self.supports_packed_decode = enable_decode
        self._packed_decode = None
        self._replayssm_decode = None
        self._chunk_kda = None
        if enable_decode or enable_prefill:
            try:
                import helion  # noqa: F401
            except ModuleNotFoundError as error:
                if error.name != "helion":
                    raise
                raise ImportError(
                    "The Helion package is required when a KDA backend is set to "
                    "Helion. Install it with: pip install helion==1.4.0"
                ) from None
        if enable_decode:
            from sglang.kernels.ops.attention.helion.kda_decode import (
                helion_fused_recurrent_kda_packed_decode,
            )
            from sglang.kernels.ops.attention.helion.kda_replayssm import (
                helion_fused_recurrent_kda_replayssm_decode,
            )

            self._packed_decode = helion_fused_recurrent_kda_packed_decode
            self._replayssm_decode = helion_fused_recurrent_kda_replayssm_decode
        if enable_prefill:
            from sglang.kernels.ops.attention.helion.kda_prefill import chunk_kda

            self._chunk_kda = chunk_kda
        self._triton = triton_fallback or TritonKDAKernel()

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
        lower_bound: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert self._packed_decode is not None
        batch_size = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(batch_size, 1, num_v_heads, head_v_dim)
        replayssm_d = kwargs.get("replayssm_d")
        replayssm_k = kwargs.get("replayssm_k")
        replayssm_g = kwargs.get("replayssm_g")
        replayssm_write_pos = kwargs.get("replayssm_write_pos")
        if (
            replayssm_d is not None
            and replayssm_k is not None
            and replayssm_g is not None
            and replayssm_write_pos is not None
        ):
            assert self._replayssm_decode is not None
            self._replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a.reshape(batch_size, num_v_heads, -1).contiguous(),
                b=b.reshape(batch_size, num_v_heads).contiguous(),
                A_log=A_log.reshape(-1),
                dt_bias=dt_bias.reshape(num_v_heads, -1).contiguous(),
                scale=scale,
                initial_state=ssm_states,
                d_cache=replayssm_d,
                k_cache=replayssm_k,
                g_cache=replayssm_g,
                out=out,
                ssm_state_indices=cache_indices,
                write_pos=replayssm_write_pos,
                force_flush=kwargs.get("replayssm_force_flush"),
                use_qk_l2norm_in_kernel=True,
                lower_bound=lower_bound,
            )
            return out.transpose(0, 1)

        if a.ndim != 2:
            a = a.reshape(batch_size, -1)
        if b.ndim != 2:
            b = b.reshape(batch_size, -1)
        self._packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            lower_bound=lower_bound,
        )
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self._triton.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        lower_bound: float | None = None,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert self._chunk_kda is not None
        return self._chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            output_intermediate_states=return_intermediate_states,
        )
