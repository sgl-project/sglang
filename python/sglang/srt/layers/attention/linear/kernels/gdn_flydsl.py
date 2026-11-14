"""AITER FlyDSL prefill kernels for GDN linear attention."""

from collections.abc import Sequence

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.utils.log_utils import rank0_log


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
        from aiter.ops.triton._triton_kernels.gated_delta_net.prefill.chunk_o import (
            chunk_fwd_o_opt_vk,
        )
        from aiter.ops.triton._triton_kernels.gated_delta_net.utils import (
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
        return o, h

    return (
        prefill,
        gdn_prepare_flydsl_supported,
        build_gated_delta_rule_prefill_metadata,
    )


class FlyDSLGDNKernel(TritonGDNKernel):
    """GDN prefill via AITER FlyDSL. Decode and verify stay on Triton."""

    _logged_triton_fallback = False
    _indices_batch = None

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
            if not self._logged_triton_fallback:
                self._logged_triton_fallback = True
                rank0_log(
                    "FlyDSL GDN prefill fell back to Triton "
                    f"(schedule={prefill_metadata is not None}, "
                    f"inplace={inplace_update})."
                )
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
        # indexed K5 path expects an in-bounds slot and writes the final state
        # in place, so send those ignored rows to the pool's padding sink, row 0
        # (MambaSlotAllocator never hands it out; the last row is a live slot).
        # The indices are fixed within a batch, so remap once per batch rather
        # than once per layer.
        if self._indices_batch is not prefill_metadata:
            self._indices_batch = prefill_metadata
            self._ssm_cache_indices = cache_indices.clamp(min=0).to(torch.int32)
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


def _load_aiter_fused_gdn():
    try:
        from aiter.ops.flydsl.gdn_fused_prefill import (
            gdn_fused_prefill_fwd,
            gdn_fused_prefill_supported,
        )
    except ImportError as exc:
        raise ImportError(
            "The fused FlyDSL GDN prefill backend requires an AITER build with "
            "aiter.ops.flydsl.gdn_fused_prefill (ROCm/aiter#6496)."
        ) from exc
    return gdn_fused_prefill_fwd, gdn_fused_prefill_supported


def _fill_track_state_from_h(
    h: torch.Tensor,
    track_state: torch.Tensor,
    track_chunk_idx: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Copy each sequence's tracked chunk-start state out of the per-chunk ``h``.

    ``h`` is [1, total_chunks, H, V, K] with each sequence's 64-token chunks
    contiguous, as both the staged FlyDSL and the Triton paths return it; rows
    with ``track_chunk_idx < 0`` receive an arbitrary chunk and are never read.
    Index math stays on the device so the copy does not sync the stream.
    """
    seq_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_chunks = (seq_lens + 63) // 64
    first_chunk = torch.cumsum(num_chunks, 0) - num_chunks
    src = (first_chunk + track_chunk_idx.clamp(min=0)).clamp(max=h.shape[1] - 1)
    track_state.copy_(h[0].index_select(0, src.long()))


class FlyDSLFusedGDNKernel(FlyDSLGDNKernel):
    """GDN prefill as one AITER FlyDSL kernel per batch: the chunk loop keeps
    the state on chip, and q/k l2norm and the fp32 radix track snapshot run
    in-kernel. Batches it declines take the staged FlyDSL path."""

    supports_track_state_snapshot = True
    _logged_staged_fallback = False
    _max_seqlen_batch = None

    def __init__(self):
        super().__init__()
        self._fused_fn, self._fused_supported = _load_aiter_fused_gdn()

    def _batch_max_seqlen(self, prefill_metadata) -> int | None:
        # AITER's speed heuristic: it declines long sequences with few heads,
        # where the staged path is faster. Cached once per batch.
        if prefill_metadata is None:
            return None
        if self._max_seqlen_batch is not prefill_metadata:
            self._max_seqlen_batch = prefill_metadata
            self._max_seqlen = max(prefill_metadata.layout.seq_lens_cpu)
        return self._max_seqlen

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
        track_state = kwargs.get("track_state")
        track_chunk_idx = kwargs.get("track_chunk_idx")
        max_seqlen = self._batch_max_seqlen(kwargs.get("prefill_metadata"))
        if inplace_update and self._fused_supported(
            q, k, v, g, beta, ssm_states, max_seqlen=max_seqlen
        ):
            # The kernel skips negative (padded) cache indices itself.
            output = self._fused_fn(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                cu_seqlens=query_start_loc,
                track_state=track_state,
                track_chunk_idx=track_chunk_idx,
            )
            return output, None, None

        if not self._logged_staged_fallback:
            self._logged_staged_fallback = True
            rank0_log(
                "Fused FlyDSL GDN prefill declined a batch; running the staged "
                f"FlyDSL path (inplace={inplace_update}, max_seqlen={max_seqlen})."
            )
        output, final_state, h = super().extend(
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
        # The backend reads the snapshot instead of h for every snapshot kernel.
        if track_state is not None:
            _fill_track_state_from_h(h, track_state, track_chunk_idx, query_start_loc)
        return output, final_state, h
