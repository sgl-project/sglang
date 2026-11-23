"""SGLang's GDN backend under TOP (Qwen3.5 GDN).

Chunk prefill runs with the fixed 64-row output tile, so its bytes do not depend on the
batch, and decode follows that prefill bit for bit (``gdn_chunked``): the trainer runs
the same kernel object's prefill over each sample.
"""

from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kernels import gdn_triton
from sglang.srt.true_on_policy.contracts import resolve_true_on_policy_runtime_policy
from sglang.srt.true_on_policy.gdn_contract import validate_gdn_contract


def top_gdn_attn_backend(runner):
    """The GDN backend for TOP runs; None when TOP is off."""
    if not resolve_true_on_policy_runtime_policy(runner.server_args).enabled:
        return None
    from sglang.srt.true_on_policy.gdn_chunked_backend import _TopGDNChunkedAttnBackend

    return _TopGDNChunkedAttnBackend(runner)


class _FixedTileGDNKernel(gdn_triton.TritonGDNKernel):
    # Radix track snapshots come from the FP32 track buffer, not the BF16 per-chunk h,
    # so a restored state equals the state a cold prefill reaches.
    supports_track_state_snapshot = True

    def extend(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        inplace_update=True,
        track_state=None,
        track_chunk_idx=None,
        **kwargs,
    ):
        return gdn_triton.chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            inplace_update=inplace_update,
            fixed_output_tile=True,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
        )


_KERNEL = None


def top_gdn_kernel():
    """The GDN chunk prefill serving runs under TOP; the trainer's prefill calls this same kernel."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _FixedTileGDNKernel()
    return _KERNEL


class _TopGDNAttnBackend(GDNAttnBackend):
    """The native backend checked against the TOP GDN contract, its kernels pinned to one object (the chunked
    backend then swaps in its own)."""

    def __init__(self, model_runner):
        from sglang.srt.arg_groups.overrides import resolved_view

        validate_gdn_contract(resolved_view(model_runner.server_args))
        super().__init__(model_runner)
        dispatcher = self.kernel_dispatcher
        if type(dispatcher.extend_kernel) is not gdn_triton.TritonGDNKernel:
            raise ValueError("GDN TOP requires the Triton chunk prefill kernel")
        if (
            type(dispatcher.decode_kernel) is not gdn_triton.TritonGDNKernel
            or not dispatcher.supports_packed_decode
        ):
            raise ValueError("GDN TOP requires the Triton packed decode dispatch")
        # One kernel object for prefill and decode; the trainer's prefill calls the same one.
        dispatcher.extend_kernel = dispatcher.decode_kernel = top_gdn_kernel()
