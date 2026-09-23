"""AITER FlyDSL prefill kernels for GDN linear attention."""

from collections.abc import Sequence

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel


def _load_aiter_flydsl_gdn():
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            chunk_gated_delta_rule_fwd_h_flydsl_opt,
            gdn_prepare_flydsl_supported,
            gdn_prepare_fwd_flydsl,
        )
        from aiter.ops.prefill_batch_metadata import (
            build_gated_delta_rule_prefill_metadata,
        )
        from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
            chunk_fwd_o_opt_vk,
        )
        from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
            l2norm_fwd,
        )
    except ImportError as exc:
        raise ImportError(
            "The FlyDSL GDN prefill backend requires the AITER version shipped "
            "with the SGLang ROCm image."
        ) from exc

    # The stages of AITER's chunk_gated_delta_rule_opt_vk, which discards the
    # per-chunk states `h` that radix-cache Mamba state tracking reads.
    def prefill(
        q, k, v, g, beta, *, o, ssm_states, cache_indices, cu_seqlens, prefill_metadata
    ):
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
        w, u, g_cumsum = gdn_prepare_fwd_flydsl(
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            use_exp2=True,
            prefill_metadata=prefill_metadata,
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_h_flydsl_opt(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=ssm_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_dtype=ssm_states.dtype,
            use_exp2=True,
            g_head_major=True,
            prefill_metadata=prefill_metadata,
            snapshot_dtype=torch.bfloat16,
            initial_state_indices=cache_indices,
            inplace_final_state=True,
        )
        if o is None:
            o = v.new_empty(v.shape)
        o = chunk_fwd_o_opt_vk(
            q=q,
            k=k,
            v=v_new,
            o=o,
            h=h,
            g=g_cumsum,
            scale=k.shape[-1] ** -0.5,
            cu_seqlens=cu_seqlens,
            use_exp2=True,
            prefill_metadata=prefill_metadata,
        )
        return o.to(q.dtype), h

    return (
        prefill,
        gdn_prepare_flydsl_supported,
        build_gated_delta_rule_prefill_metadata,
    )


class FlyDSLGDNKernel(TritonGDNKernel):
    """GDN prefill via AITER FlyDSL. Decode and verify stay on Triton."""

    def __init__(self):
        (
            self._prefill_fn,
            self._prepare_supported,
            self._metadata_builder,
        ) = _load_aiter_flydsl_gdn()

    def build_prefill_metadata(
        self,
        seq_lens_cpu: Sequence[int],
        *,
        cu_seqlens: torch.Tensor,
    ):
        return self._metadata_builder(
            seq_lens_cpu,
            cu_seqlens=cu_seqlens,
            chunk_size=64,
        )

    def _supports_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ssm_states: torch.Tensor,
    ) -> bool:
        # The prepare check also covers bf16 K/V with head dim 128.
        return (
            q.dtype == torch.bfloat16
            and ssm_states.dtype in (torch.bfloat16, torch.float32)
            and self._prepare_supported(k, v)
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
        inplace_update: bool = True,
        **kwargs,
    ) -> tuple:
        prefill_metadata = kwargs.get("prefill_metadata")
        if (
            prefill_metadata is None
            or not inplace_update
            or not self._supports_inputs(q, k, v, ssm_states)
        ):
            return super().extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                inplace_update=inplace_update,
                **kwargs,
            )

        # DP-attention MLP-sync rows use -1 as a padding marker. AITER's
        # indexed K5 path expects an in-bounds slot, so send those ignored rows
        # to the pool's reserved sentinel slot. The indices are fixed within a
        # batch, so remap once per batch rather than once per layer.
        if getattr(self, "_indices_batch", None) is not prefill_metadata:
            self._indices_batch = prefill_metadata
            self._ssm_cache_indices = torch.where(
                cache_indices >= 0,
                cache_indices,
                ssm_states.shape[0] - 1,
            ).to(torch.int32)
        output, h = self._prefill_fn(
            q,
            k,
            v,
            g,
            beta,
            o=kwargs.get("output"),
            ssm_states=ssm_states,
            cache_indices=self._ssm_cache_indices,
            cu_seqlens=query_start_loc,
            prefill_metadata=prefill_metadata,
        )
        return output, None, h
