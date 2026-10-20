"""Non-CUDA DeepSeek V4 decoder and draft execution.

The mixins own platform mHC state, fusion, and TBO operations.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    can_fuse_all_reduce,
)
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
    _gfx95_dense_post_attention_norm,
    apply_attention_mhc,
    apply_mhc_post_pre_boundary,
    apply_moe_mhc,
    attention_mhc_fusion,
)
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
    hc_boundary as make_amd_hc_boundary,
)
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
    moe_mhc_fusion,
)
from sglang.srt.models.deepseek_common.moe_transport import (
    MoeRowTransport,
    _tbo_combine_moe_output,
    _tbo_gather_moe_input,
    _tbo_run_moe,
    _tbo_wait_moe_output,
    shared_expert_runs_locally,
)
from sglang.srt.models.deepseek_v2 import MoEOutput
from sglang.srt.models.deepseek_v4_mhc import MHC_POST_MULT_VALUE, AttnOutput
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models import deepseek_v2

_is_gfx95_supported = is_gfx95_supported()


def _prepare_ffn_input(layer, hidden_states, residual, post, comb, forward_batch):
    """Close attention and prepare the FFN with the platform's mHC policy."""
    if layer.use_fused_mhc_post_pre:
        post_attn_norm_weight = (
            layer._post_attention_layernorm_weight_bf16
            if layer._post_attention_layernorm_weight_bf16 is not None
            else layer.post_attention_layernorm.weight.data
        )
        fused = apply_mhc_post_pre_boundary(
            hidden_states,
            residual,
            post,
            comb,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.hc_mult,
            layer.rms_norm_eps,
            layer.hc_eps,
            MHC_POST_MULT_VALUE,
            layer.hc_sinkhorn_iters,
            post_attn_norm_weight,
            layer.post_attention_layernorm.variance_epsilon,
            fn_transpose=True,
        )
        if fused is not None:
            residual, hidden_states, post, comb, norm_fused = fused
            if not norm_fused:
                hidden_states = layer.post_attention_layernorm(hidden_states)
            return residual, hidden_states, post, comb
    hidden_states = layer.hc_post(hidden_states, residual, post, comb)
    residual = hidden_states
    hidden_states, post, comb, norm_fused = layer.hc_pre(
        hidden_states,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        norm=layer.post_attention_layernorm,
        forward_batch=forward_batch,
    )
    if not norm_fused:
        hidden_states = layer.post_attention_layernorm(hidden_states)

    return residual, hidden_states, post, comb


class NonCudaDecoderMixin:
    """Platform tuple and HcState execution for target and NextN decoders.

    CUDA mHC completion lives in deepseek_v4_boundary; operand movement and
    finalization live in deepseek_common.moe_transport. Changes to completion
    ordering must be checked in both implementations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_layer = None
        self.local_boundary = None
        self.next_boundary = None

    def bind_next(self, next_layer):
        if not self.hc_pre_from_prev_sublayer:
            return
        self._next_layer = None
        if isinstance(next_layer, NonCudaDecoderMixin) and next_layer.engram is None:
            self._next_layer = next_layer

    def refresh_mhc_norm_weight_cache(self):
        super().refresh_mhc_norm_weight_cache()
        self.local_boundary = None
        self.next_boundary = None

    @cached_property
    def _can_fuse_attn_mhc(self):
        return (
            get_parallel().tp_size == self.self_attn.attn_tp_size == 4
            and self.self_attn.wo_b.reduce_results
            and self.local_boundary.norm_fusable
            and mhc.can_fuse_post(self.hc_cfg)
        )

    def forward_hc_pre_from_prev(
        self,
        positions: torch.Tensor,
        state: mhc.HcState,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
    ) -> mhc.HcState:
        """The layer's two hyper-connections, each collapsing with the previous one's
        pre-mix."""
        self._init_boundaries()
        stats_stream = None
        if mhc.use_stats_stream(self.hc_cfg, forward_batch, state.residual):
            stats_stream = self.hc_stats_stream

        def run_attn_hc(state: mhc.HcState) -> mhc.HcState:
            assert self.attn_hc is not None
            residual = state.residual
            quantized = [] if self.self_attn.accepts_mxfp8_swizzled_input() else None
            mhc.fork_stats_stream(stats_stream)
            x = mhc.combine(self.attn_hc, state, quantized)
            precomputed = state.stats
            state.release()
            del state
            world_size = self.self_attn.attn_tp_size
            fuse_all_reduce_mhc = self._can_fuse_attn_mhc and can_fuse_all_reduce(
                x.shape[0], self.hc_cfg.hidden
            )
            with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
                y = self.self_attn(
                    x=x,
                    positions=positions,
                    forward_batch=forward_batch,
                    x_quant=quantized[0] if quantized else None,
                    defer_all_reduce=fuse_all_reduce_mhc,
                )
            del x
            return mhc.run_attn_post(
                self.attn_hc,
                y,
                residual,
                stats_stream=stats_stream,
                next=self.local_boundary,
                world_size=world_size,
                precomputed=precomputed,
            )

        def run_ffn_hc(state: mhc.HcState) -> mhc.HcState:
            assert self.ffn_hc is not None
            residual = state.residual
            mhc.fork_stats_stream(stats_stream)
            x = mhc.combine(self.ffn_hc, state)
            precomputed = state.stats
            state.release()
            del state
            fuse_all_reduce_mhc = self._can_fuse_ffn_mhc and can_fuse_all_reduce(
                x.shape[0], self.hc_cfg.hidden
            )
            y = self._run_moe_ffn_dp_sync(
                x,
                forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
                return_moe_output=fuse_all_reduce_mhc,
            )
            del x
            return mhc.run_moe_post(
                self.ffn_hc,
                y,
                residual,
                stats_stream=stats_stream,
                next=self.next_boundary,
                world_size=self.mlp.tp_size,
                precomputed=precomputed,
            )

        return run_ffn_hc(run_attn_hc(state))

    def _init_boundaries(self) -> None:
        """Resolved on first forward: `accepts_mxfp8_swizzled_input` is not settled
        until `process_weights_after_loading`."""
        if self.local_boundary is not None:
            return
        # ATTN[i] -> MOE[i]
        self.local_boundary = mhc.make_boundary(
            self.post_attention_layernorm,
            accepts_mxfp8=False,
            hc=self.ffn_hc,
        )
        # MOE[i] -> ATTN[i+1]
        if self._next_layer is not None:
            self.next_boundary = mhc.make_boundary(
                self._next_layer.input_layernorm,
                accepts_mxfp8=self._next_layer.self_attn.accepts_mxfp8_swizzled_input(),
                hc=self._next_layer.attn_hc,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
        prev_residual: Optional[torch.Tensor] = None,
        prev_post: Optional[torch.Tensor] = None,
        prev_comb: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        use_fused = self.use_fused_mhc_post_pre

        fused = None
        if prev_residual is not None and use_fused:
            # Dispatch cascade: aiter HIP (gfx95) -> Triton (gfx95 small-batch
            # <=64 tokens, or gfx1250 all sizes) -> TileLang -> None.
            input_norm_weight = (
                self._input_layernorm_weight_bf16
                if self._input_layernorm_weight_bf16 is not None
                else self.input_layernorm.weight.data
            )
            fused = apply_mhc_post_pre_boundary(
                hidden_states,
                prev_residual,
                prev_post,
                prev_comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.hc_mult,
                self.rms_norm_eps,
                self.hc_eps,
                MHC_POST_MULT_VALUE,
                self.hc_sinkhorn_iters,
                input_norm_weight,
                self.input_layernorm.variance_epsilon,
                fn_transpose=False,
            )
        if fused is not None:
            residual, hidden_states, post, comb, norm_fused = fused
        else:
            if prev_residual is not None and use_fused:
                hidden_states = self.hc_post(
                    hidden_states, prev_residual, prev_post, prev_comb
                )
            residual = hidden_states
            hidden_states, post, comb, norm_fused = self.hc_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm=self.input_layernorm,
                forward_batch=forward_batch,
            )
        if not norm_fused:
            # Triton post+pre can leave norm separate: gfx95 may quantize it,
            # while gfx1250 must keep the plain layernorm path.
            hidden_states, x_quant = self._input_norm(
                hidden_states, allow_aiter_quant=_is_gfx95_supported
            )
        else:
            x_quant = None

        with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
            hidden_states = self.self_attn(
                x=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                x_quant=x_quant,
            )

        del fused
        residual, hidden_states, post, comb = _prepare_ffn_input(
            self, hidden_states, residual, post, comb, forward_batch
        )

        hidden_states = self._run_moe_ffn_dp_sync(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )

        if not use_fused:
            hidden_states = self.hc_post(hidden_states, residual, post, comb)
            return hidden_states, None, None, None

        # Return the deferred FFN hc_post state; the next layer consumes it with
        # cross-layer fusion, and the final layer is completed in DeepseekV4Model.
        return hidden_states, residual, post, comb

    def _run_moe_ffn_dp_sync(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        input_ids: Optional[torch.Tensor],
        input_ids_global: Optional[torch.Tensor],
        return_moe_output: bool = False,
    ) -> Union[torch.Tensor, deepseek_v2.MoEOutput]:
        prepared = MoeRowTransport(
            self.mlp,
            hidden_states,
            forward_batch,
            input_ids,
            input_ids_global,
            cp_return_contiguous=self.config.model_type == "deepseek_v41",
        )
        with prepared.scope():
            return_moe_output = (
                return_moe_output
                and not prepared._use_cp
                and not prepared._use_tp_moe_gather
                and not prepared._use_tp_attn_a2a_scatter
            )
            hidden_states = self.mlp(
                prepared.input,
                forward_batch,
                input_ids=prepared.input_ids,
                input_ids_global=prepared.input_ids_global,
                skip_shared_experts=prepared.do_shared_local,
                return_moe_output=return_moe_output,
            )
        prepared.input = None
        return prepared.return_output(hidden_states)

    def op_mhc_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
        tbo_subbatch_index: Optional[int] = None,
        **kwargs,
    ):
        # Non-fused attention-side mHC pre + input layernorm.
        attn_residual = hidden_states
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            norm=self.input_layernorm,
            forward_batch=forward_batch,
        )
        if not norm_fused:
            hidden_states, x_quant = self._input_norm(hidden_states)
        else:
            x_quant = None

        state.attn_residual = attn_residual
        state.attn_post = post
        state.attn_comb = comb
        state.hidden_states_after_input_norm = hidden_states
        state.attn_x_quant = x_quant
        # mori's op_output slices final_hidden_states[:num_tokens].
        if get_moe_a2a_backend().is_mori():
            state.num_tokens = attn_residual.shape[0]
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_mhc_post_attn_pre_mlp(self, state):
        # Close the attention mHC (hc_post), then open the FFN-side mHC pre +
        # post-attention layernorm. Produces the 2D MoE input.
        #
        # Pop each boundary tensor from the state EXACTLY ONCE, up front, and
        # reuse the locals for both the fused attempt and the non-fused
        # fallback. apply_mhc_post_pre_boundary() returns None when it declines
        # to fuse -- most importantly for the 0-token DP two-batch-overlap idle
        # ubatch -- in which case control must fall through to the unfused
        # hc_post. Popping in the fused call's arguments and again in the
        # fallback would double-pop -> KeyError: 'hidden_states_after_attn' on
        # every idle DP rank. use_fused_mhc_post_pre is on whenever the aiter
        # gfx95 mHC path is available (is_cross_layer_mhc_fusion_enabled), so
        # this fallback is reached under DP regardless of the TileLang env.
        hidden_states_after_attn = state.pop("hidden_states_after_attn")
        attn_residual = state.pop("attn_residual")
        attn_post = state.pop("attn_post")
        attn_comb = state.pop("attn_comb")

        residual, hidden_states, post, comb = _prepare_ffn_input(
            self,
            hidden_states_after_attn,
            attn_residual,
            attn_post,
            attn_comb,
            state.forward_batch,
        )
        state.ffn_residual = residual
        state.ffn_post = post
        state.ffn_comb = comb
        state.hidden_states_mlp_input = hidden_states

    def op_mhc_postprocess(self, state):
        # Close the FFN mHC (hc_post) and emit the next layer's input dict.
        hidden_states = self.hc_post(
            state.pop("hidden_states_mlp_output"),
            state.pop("ffn_residual"),
            state.pop("ffn_post"),
            state.pop("ffn_comb"),
        )
        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            # DSV4 non-fused layers carry no residual across layers; the key is
            # required by the next layer's op_mhc_prepare_attn (ignored).
            residual=None,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )
        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output

    def op_gather_a(self, state):
        local = state.pop("hidden_states_mlp_input")
        state.do_shared_local = shared_expert_runs_locally(self.mlp)
        state.global_hidden, state.gather_event, state.shared_local = (
            _tbo_gather_moe_input(
                self.mlp,
                local,
                state.forward_batch,
                state.tbo_subbatch_index,
                state.do_shared_local,
            )
        )
        state.gather_keepalive = local

    def op_gather_b(self, state):
        torch.cuda.current_stream().wait_event(state.pop("gather_event"))
        # Compute now ordered after the gather -> the gather input is safe to
        # release (freed on the compute stream, no record_stream deferral).
        state.pop("gather_keepalive")

    def op_moe(self, state):
        state.global_expert_out = _tbo_run_moe(
            self, state.pop("global_hidden"), state.forward_batch, state.do_shared_local
        )

    def op_combine_a(self, state):
        output = state.pop("global_expert_out")
        state.local_out, state.combine_event = _tbo_combine_moe_output(
            output, state.tbo_subbatch_index
        )
        state.combine_keepalive = output

    def op_combine_b(self, state):
        state.hidden_states_mlp_output = _tbo_wait_moe_output(
            state.pop("local_out"),
            state.pop("combine_event"),
            state.pop("shared_local"),
        )
        state.pop("combine_keepalive")
        state.pop("do_shared_local")


class NonCudaDSparkMixin(NonCudaDecoderMixin):
    """Draft stage execution with plain norms and its attention call."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_pre: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.hc_pre_from_prev_sublayer:
            return self._forward_hc_pre_from_prev(
                positions, hidden_states, forward_batch, prev_pre
            )
        residual = hidden_states
        x, post, comb = self._hc_pre_block(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
            x = self.self_attn(positions, x, forward_batch)
        x = self._hc_post_block(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre_block(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.post_attention_layernorm(x)
        x = self._run_ffn(x, forward_batch)
        x = self._hc_post_block(x, residual, post, comb)
        return x, None

    def _hc_pre_block(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, post, comb, _ = self.hc_pre(x, hc_fn, hc_scale, hc_base)
        return y, post, comb

    def _hc_post_block(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return self.hc_post(x, residual, post, comb)

    def _forward_hc_pre_from_prev(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_pre: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # The draft stages run the plain four steps: no fused post, no seams, so
        # they need nothing from `bind_next` and never hand anything forward.
        assert self.attn_hc is not None and self.ffn_hc is not None
        stats_stream = None
        if mhc.use_stats_stream(self.hc_cfg, forward_batch, hidden_states):
            stats_stream = self.hc_stats_stream

        residual = hidden_states
        mhc.fork_stats_stream(stats_stream)
        x = mhc.combine(self.attn_hc, mhc.HcState(hidden_states, prev_pre))
        with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
            x = self.self_attn(positions, x, forward_batch)
        attn_pre, attn_post, attn_comb = mhc.mix_stats(
            self.attn_hc, hidden_states, stats_stream
        )
        if stats_stream is not None:
            torch.cuda.current_stream().wait_stream(stats_stream)
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden_states
        mhc.fork_stats_stream(stats_stream)
        x = mhc.combine(self.ffn_hc, mhc.HcState(hidden_states, attn_pre))
        x = self._run_ffn(x, forward_batch)
        ffn_pre, ffn_post, ffn_comb = mhc.mix_stats(
            self.ffn_hc, hidden_states, stats_stream
        )
        if stats_stream is not None:
            torch.cuda.current_stream().wait_stream(stats_stream)
        hidden_states = self.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre

    def _run_ffn(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        y = self._run_moe_ffn_dp_sync(
            x, forward_batch, input_ids=None, input_ids_global=None
        )
        return y.view(shape)


def forward_layer_fused_boundary(
    model,
    i: int,
    state,
    *,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    forward_batch,
    input_ids_global: torch.Tensor,
    capture_dspark: bool,
):
    """Layer i through the fused mHC boundary. Its FFN hc_post stays pending for the next
    layer's boundary launch unless that layer reads the residual stream first (Engram gate,
    DSpark capture) or there is none; hidden_states is None while a post is pending."""
    nxt = i + 1
    defer_post = (
        nxt < model.end_layer
        and model.layers[nxt].engram is None
        and not (capture_dspark and nxt in model.dspark_layers_to_capture)
    )
    # The boundary speaks its own (hidden_states, prev_pre, pending_post) triple;
    # the loop speaks HcState. This is the only place the two meet.
    from sglang.srt.models.deepseek_v4_mhc import HcPending, HcState

    pending = isinstance(state.streams, HcPending)
    hidden_states, prev_pre, pending_post = forward_hc_pre_from_prev_fused_boundary(
        model.layers[i],
        positions=positions,
        hidden_states=None if pending else state.streams,
        input_ids=input_ids,
        forward_batch=forward_batch,
        input_ids_global=input_ids_global,
        prev_pre=state.pre,
        pending_post=tuple(state.streams) if pending else None,
        defer_post=defer_post,
    )
    return HcState(
        hidden_states if pending_post is None else HcPending(*pending_post),
        prev_pre,
    )


def fused_boundary_attn_input(layer, state: mhc.HcState) -> torch.Tensor:
    """
    Compute the previous layer's mHC post and this layer's attention input for every
    row, with the kernel that starts forward_hc_pre_from_prev_fused_boundary.
    """
    if isinstance(state.streams, mhc.HcPending):
        pending_post = tuple(state.streams)
    else:
        pending_post = (None, state.streams, None, None)
    _, x, coefficients = make_amd_hc_boundary(
        layer,
        *pending_post,
        state.pre,
        layer.hc_attn_fn,
        layer.hc_attn_scale,
        layer.hc_attn_base,
    )
    x, _ = layer._input_norm(x, allow_aiter_quant=False, coefficients=coefficients)
    return x


def forward_hc_pre_from_prev_fused_boundary(
    layer,
    positions: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    forward_batch,
    input_ids_global: torch.Tensor,
    prev_pre: Optional[torch.Tensor],
    pending_post: Optional[Tuple[torch.Tensor, ...]],
    defer_post: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
    """ROCm fused attention and FFN hyper-connections.
    pending_post is the previous layer's unapplied FFN hc_post (x, residual, post,
    comb); with defer_post this layer's is returned the same way and hidden_states is None."""
    if pending_post is None:
        pending_post = (None, hidden_states, None, None)
    residual, x, attn_coefficients = make_amd_hc_boundary(
        layer,
        *pending_post,
        prev_pre,
        layer.hc_attn_fn,
        layer.hc_attn_scale,
        layer.hc_attn_base,
    )
    # the boundary's reduce + sinkhorn rides in the norm launch
    x, x_quant = layer._input_norm(
        x, allow_aiter_quant=False, coefficients=attn_coefficients
    )
    with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
        mhc = attention_mhc_fusion(layer, residual, attn_coefficients, forward_batch)
        x = layer.self_attn(
            x=x,
            positions=positions,
            forward_batch=forward_batch,
            x_quant=x_quant,
            defer_all_reduce=mhc is not None,
        )
    if mhc is not None:
        # wo_b honors the deferred reduce under attention_mhc_fusion's gates;
        # fail loud rather than fall back to an unfused reduce.
        assert isinstance(x, AttnOutput)
        residual = apply_attention_mhc(x.partial, mhc)
        x = None
    residual, x, ffn_coefficients = make_amd_hc_boundary(
        layer,
        x,
        residual,
        attn_coefficients.post if x is not None else None,
        attn_coefficients.comb if x is not None else None,
        attn_coefficients.pre,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
    )
    x = _gfx95_dense_post_attention_norm(layer, x, ffn_coefficients)
    mhc = moe_mhc_fusion(layer, residual, ffn_coefficients, forward_batch)
    x = layer._run_moe_ffn_dp_sync(
        x,
        forward_batch,
        input_ids=input_ids,
        input_ids_global=input_ids_global,
        return_moe_output=mhc is not None,
    )
    ffn_pre, ffn_post, ffn_comb = ffn_coefficients.tensors()
    if mhc is not None:
        # The MoE TP path honors the deferred output under moe_mhc_fusion's gates;
        # fail loud rather than fall back to an unfused reduce. The fused post
        # carries the reduce, so take the merged rows without one.
        assert isinstance(x, MoEOutput)
        # Reduction already applied post. The next boundary consumes this
        # materialized residual, including when it would normally defer post.
        return apply_moe_mhc(x.get_merged(), mhc), ffn_pre, None
    if defer_post:
        return None, ffn_pre, (x, residual, ffn_post, ffn_comb)
    return layer.hc_post(x, residual, ffn_post, ffn_comb), ffn_pre, None
