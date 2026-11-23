"""SGLang backend of TOP GDN (see gdn_chunked.py): adapts forward metadata to ``ChunkedGDNEngine``.

The backend keeps the native GDN backend's convolution, gating and radix-track code and swaps the kernel object
its dispatcher calls for decode, extend and verify. Per forward it builds the engine's coordinates once: for decode
and verify, device tensors (static per batch size under CUDA graphs, refreshed in ``init_forward_metadata_out_graph``
with device ops only, so the metadata glue graph may capture them); for an extend, a host plan.
"""

import dataclasses

import torch

from sglang.srt.layers.attention.linear.kernels import gdn_triton
from sglang.srt.true_on_policy.gdn_backend import _TopGDNAttnBackend
from sglang.srt.true_on_policy.gdn_chunked_engine import (
    ChunkedGDNEngine,
    strided_mixed,
)


class _ChunkedGDNKernel(gdn_triton.TritonGDNKernel):
    """The dispatcher's decode, extend and verify kernel under TOP chunked GDN; the layer and the forward's
    coordinates come from the backend."""

    supports_track_state_snapshot = True
    supports_strided_target_verify_qkv = True
    # Verify writes neither E nor per-draft states: the pool allocates no intermediate_ssm.
    verify_needs_intermediate_states = False

    def __init__(self, backend: "_TopGDNChunkedAttnBackend"):
        self.backend = backend

    def packed_decode(
        self,
        mixed_qkv,
        a,
        b,
        *,
        A_log,
        dt_bias,
        scale,
        ssm_states,
        cache_indices,
        num_v_heads,
        head_v_dim,
        **kwargs,
    ):
        be = self.backend
        out = be.engine.decode(
            be.layer, be.step, mixed_qkv, a, b, A_log, dt_bias, ssm_states
        )
        return out.unsqueeze(0)

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
        track_state=None,
        track_chunk_idx=None,
        **kwargs,
    ):
        be = self.backend
        a, b, A_log, dt_bias = be.gates
        out = be.engine.extend(
            be.layer,
            be.plan,
            q,
            k,
            v,
            g,
            beta,
            a,
            b,
            A_log,
            dt_bias,
            ssm_states,
            cache_indices,
            track_state,
        )
        # (output, final state, per-chunk h): the track reads track_state, never h.
        return out, None, None

    def target_verify(
        self,
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        retrieve_parent_token=None,
        **kwargs,
    ):
        if retrieve_parent_token is not None:
            raise ValueError("TOP chunked GDN verifies chain drafts only")
        be = self.backend
        out = be.engine.verify(
            be.layer, be.step, strided_mixed(q, k, v), a, b, A_log, dt_bias, ssm_states
        )
        return out.unsqueeze(0)


class _TopGDNChunkedAttnBackend(_TopGDNAttnBackend):
    def __init__(self, model_runner):
        from sglang.srt.arg_groups.overrides import resolved_view
        from sglang.srt.true_on_policy.gdn_chunked import row_cache_bytes_per_request
        from sglang.srt.utils import rank0_log

        super().__init__(model_runner)
        args = resolved_view(model_runner.server_args)
        config = model_runner.model_config.hf_text_config
        Hg, K = config.linear_num_key_heads, config.linear_key_head_dim
        H, V = config.linear_num_value_heads, config.linear_value_head_dim
        temporal = self.req_to_token_pool.mamba_pool.mamba_cache.temporal
        if tuple(temporal.shape[-3:]) != (H, V, K):
            raise ValueError(
                f"TOP chunked GDN expects [H, V, K] = {(H, V, K)} FP32 states, got {tuple(temporal.shape)}"
            )
        spec = args.speculative_algorithm is not None
        # One row-cache entry per req-pool row; row 0 is SGLang's padding row.
        num_slots = self.req_to_token_pool.size + 1
        self.engine = ChunkedGDNEngine(
            num_layers=temporal.shape[0],
            num_slots=num_slots,
            Hg=Hg,
            H=H,
            K=K,
            V=V,
            device=self.device,
            spec=spec,
            nmax=args.speculative_num_draft_tokens if spec else 1,
            max_bs=num_slots,
        )
        per_request = row_cache_bytes_per_request(config, temporal.shape[0])
        assert self.engine.rows.nbytes() == per_request * num_slots
        self.step = self.plan = self.layer = self.gates = None
        kernel = _ChunkedGDNKernel(self)
        dispatcher = self.kernel_dispatcher
        dispatcher.extend_kernel = dispatcher.decode_kernel = (
            dispatcher.verify_kernel
        ) = kernel
        rank0_log(
            f"[top] chunked GDN: row cache {per_request / (1 << 20):.2f} MiB per request over "
            f"{temporal.shape[0]} layers x {num_slots} request rows = "
            f"{self.engine.rows.nbytes() / (1 << 30):.2f} GiB; kkt config {self.engine.kkt}; decode through "
            f"{'chunk_rows2' if spec else 'chunk_rows'}"
        )

    # ------------------------------------------------------------------ metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self.engine.alloc_graph_bufs(max_bs)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self.plan = None
        # A captured graph replays its kernels whatever the batch, so an idle replay (attention DP) refreshes its
        # coordinates too: padding rows (mamba slot -1) write only the padding request row.
        self.step = self._build_step(forward_batch, static=True)

    def init_forward_metadata(self, forward_batch):
        super().init_forward_metadata(forward_batch)
        self.step = self.plan = None
        mode = forward_batch.forward_mode
        if mode.is_idle() or forward_batch.batch_size == 0:
            return
        if mode.is_decode() or mode.is_target_verify():
            self.step = self._build_step(forward_batch, static=False)
        elif mode.is_extend():
            self.plan = self._build_extend_plan(forward_batch)

    def _build_step(self, forward_batch, *, static: bool):
        bs = forward_batch.batch_size
        nmax = 1
        if forward_batch.forward_mode.is_target_verify():
            spec = forward_batch.spec_info
            if getattr(spec, "ragged_verify_layout", None) is not None:
                raise ValueError(
                    "TOP chunked GDN verifies a dense [requests, draft tokens] window"
                )
            nmax = spec.draft_token_num
            if nmax != self.engine.nmax:
                raise ValueError(
                    f"TOP chunked GDN verify got {nmax} rows per request, sized for {self.engine.nmax}"
                )
        return self.engine.build_step(
            positions=forward_batch.positions,
            ring=forward_batch.req_pool_indices,
            state=self.forward_metadata.mamba_cache_indices,
            bs=bs,
            nmax=nmax,
            static=static,
        )

    def _build_extend_plan(self, forward_batch):
        prefix = forward_batch.extend_prefix_lens_cpu
        lens = forward_batch.extend_seq_lens_cpu
        if prefix is None or lens is None:
            raise ValueError(
                "TOP chunked GDN extends need host prefix and extend lengths"
            )
        return self.engine.build_extend_plan(
            prefix=[int(x) for x in prefix],
            lens=[int(x) for x in lens],
            ring=forward_batch.req_pool_indices.tolist(),
            state=self.forward_metadata.mamba_cache_indices.tolist(),
            track_mask=forward_batch.mamba_prefill_track_mask_cpu,
            track_lens=forward_batch.mamba_track_seqlens_cpu,
        )

    # ------------------------------------------------------------------ forwards

    def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        self.layer = self.req_to_token_pool.mamba2_layer_index(layer.layer_id)
        return super().forward_decode(layer, forward_batch, mixed_qkv, a, b, **kwargs)

    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        self.layer = self.req_to_token_pool.mamba2_layer_index(layer.layer_id)
        self.gates = (a, b, layer.A_log, layer.dt_bias)
        try:
            return super().forward_extend(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        finally:
            self.gates = None

    # ------------------------------------------------------------------ radix track and spec commit

    def _track_mamba_state_decode(
        self, forward_batch, conv_states, ssm_states, cache_indices, layer_id=None
    ):
        # A decode track fires when a request's token lands on row 63. chunk_rows closes the chunk in place there,
        # so the slot holds the boundary state the copy takes; under speculation chunk_rows2 defers the close, so
        # the copied E is replaced by the closed state, for every layer at the last one.
        super()._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer_id
        )
        engine = self.engine
        if (
            not engine.spec
            or forward_batch.mamba_track_mask is None
            or self.step is None
            or self.layer != engine.num_layers - 1
        ):
            return
        step = self.step
        bs = step.ring.numel()
        engine.close_to(
            state=self.req_to_token_pool.mamba_pool.mamba_cache.temporal,
            ring=step.ring,
            slots=step.state,
            dst=torch.where(
                forward_batch.mamba_track_mask[:bs],
                self.forward_metadata.mamba_track_indices[:bs].to(torch.int64),
                -1,
            ),
        )

    def commit_after_verify(
        self,
        *,
        state_indices,
        req_pool_indices,
        last_correct_step_indices,
        mamba_track_indices,
        mamba_steps_to_track,
        mamba_caches,
    ):
        """The verify commit: E and the row cache already hold the committed rows (rejected rows are overwritten by
        later steps), so only conv windows move; a track slot crossed this step (its chunk ended on a committed
        row 63) gets the closed chunk's state."""
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            scatter_mamba_states_after_mtp_verify,
        )

        # An empty temporal view: the scatter skips SSM states and moves the conv windows only.
        scatter_mamba_states_after_mtp_verify(
            dataclasses.replace(mamba_caches, temporal=mamba_caches.temporal[:0]),
            state_indices,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )
        if mamba_track_indices is None:
            return
        self.engine.close_to(
            state=mamba_caches.temporal,
            ring=req_pool_indices,
            slots=state_indices,
            dst=torch.where(
                mamba_steps_to_track >= 0, mamba_track_indices.to(torch.int64), -1
            ),
        )
