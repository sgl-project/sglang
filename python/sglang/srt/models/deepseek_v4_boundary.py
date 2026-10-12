"""DeepSeek hyper-connection reads, updates and operand row transport."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial

import msgspec
import torch

from sglang.srt.layers.layer_boundary.contracts import BatchVariant, ReadoutFusion
from sglang.srt.layers.layer_boundary.factories import declare_attn, declare_ffn
from sglang.srt.layers.layer_boundary.facts import ReadFacts, UpdateFacts
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    _sum_group,
)
from sglang.srt.layers.layer_boundary.output import DeferredFinalize, UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import PendingUpdateRead
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.row_transport import (
    ComputeTransport,
    RowTransport,
    TransportRows,
)
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.runtime_context import get_forward, get_parallel


@dataclass(frozen=True)
class LayerCall:
    positions: object
    input_ids: object = None
    input_ids_global: object = None
    seam_open: bool = True
    mega_seam_open: bool = True
    tbo: bool = False
    tbo_subbatch_index: object = None


class ReadState(msgspec.Struct):
    """The source residual and coefficients of one sublayer invocation."""

    streams: torch.Tensor
    post: object = None
    comb: object = None
    quantized: object = None
    stats: object = None
    stats_stream: object = None
    call: object = None


class HcUpdate:
    bind = None
    replay_update = None
    is_plain_add = False
    outlives_layer = True
    quantized_sum = False

    def __init__(self, layer, attention, *, applied_at_exit=False):
        self.layer = layer
        self.attention = attention
        self.applied_at_exit = applied_at_exit
        self.writes_stream = False

    def slice_residual_attn_tp(self, residual):
        raise NotImplementedError("mHC coefficients remain on their source rows")

    def gather_residual_attn_tp(self, residual):
        raise NotImplementedError("mHC coefficients remain on their source rows")

    def post(self, output, state, *, next=None):
        if not self.layer.boundary_pre_from_prev:
            return self.layer.hc_post(output, state.streams, state.post, state.comb)
        hc = self.layer.attn_hc if self.attention else self.layer.ffn_hc
        mega = self.layer.boundary_mhc_fusions and mhc.can_use_mega_mhc_prefill(
            self.layer.hc_cfg, seam_open=state.call.mega_seam_open
        )
        if self.attention:
            return mhc.run_attn_post(
                hc,
                output,
                state.streams,
                stats_stream=state.stats_stream,
                next=next,
                world_size=self.layer.self_attn.attn_tp_size,
                precomputed=state.stats,
                mega_mhc=mega,
            )
        return mhc.run_moe_post(
            hc,
            output,
            state.streams,
            stats_stream=state.stats_stream,
            next=next if state.call.seam_open else None,
            world_size=self.layer.mlp.tp_size,
            precomputed=state.stats,
            mega_next=next if mega else None,
        )

    def update(self, output, state):
        return self.post(output, state)


class HcRead:
    is_plain_norm = False
    reads_before_dp_gather = True
    reads_after_attn_tp_gather = True
    gathering_reads = ()

    def __init__(self, layer, attention):
        self.layer = layer
        self.attention = attention
        self.completing_fusions = tuple(
            ReadoutFusion(
                group,
                self._complete_read,
                reads=True,
                carried=True,
                accepts_update=lambda update: isinstance(update, HcUpdate),
            )
            for group in (SumGroup.ATTN_TP, SumGroup.MOE_OUTPUT)
        )

    def bind(self, batch, call):
        return HcReadCall(self, batch, call)

    @staticmethod
    def init_residual(hidden_states):
        return hidden_states

    def _complete_read(self, output, state, batch, norm, *, update, read_context=None):
        if not self.layer.boundary_pre_from_prev:
            return None
        if isinstance(output, MoeFinalize):
            if output.pieces is None:
                return None
            output = output.pieces
        elif isinstance(output, UnreducedOutput):
            if not update.attention:
                return None
            output = mhc.AttnOutput(output.partial)
        else:
            return None
        call = read_context if read_context is not None else state.call
        read = self.bind(batch, call)
        return read._after_post(
            update.post(output, state, next=read.next_boundary), norm
        )


class HcReadCall(HcRead):
    def __init__(self, spec, batch, call):
        self.spec = spec
        self.layer = spec.layer
        self.attention = spec.attention
        self.batch = batch
        self.call = call
        self.completing_fusions = tuple(
            msgspec.structs.replace(
                f, run=partial(spec._complete_read, read_context=call)
            )
            for f in spec.completing_fusions
        )

    def bind(self, batch, call=None):
        if batch is not self.batch or (call is not None and call is not self.call):
            raise ValueError("an invocation read cannot bind another call")
        return self

    @property
    def next_boundary(self):
        if not self.layer.boundary_mhc_fusions or (
            self.attention and self.layer.engram is not None
        ):
            return None
        hc = self.layer.attn_hc if self.attention else self.layer.ffn_hc
        return mhc.make_boundary(
            self.layer.input_layernorm
            if self.attention
            else self.layer.post_attention_layernorm,
            accepts_mxfp8=self.attention
            and self.layer.self_attn.accepts_mxfp8_swizzled_input(),
            hc=hc,
        )

    def _after_post(self, residual, norm):
        return self.read(residual, norm)

    def read(
        self,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
        *,
        pending=False,
    ):
        if quant_format or post_residual_addition is not None:
            raise NotImplementedError("mHC read supplies its own projection format")
        layer = self.layer
        if layer.boundary_pre_from_prev:
            if not isinstance(residual, mhc.HcState):
                residual = mhc.HcState(residual)
            streams = residual.residual
            stream = (
                layer.hc_stats_stream
                if mhc.use_stats_stream(layer.hc_cfg, self.batch, streams)
                else None
            )
            mhc.fork_stats_stream(stream)
            quantized = (
                []
                if self.attention and layer.self_attn.accepts_mxfp8_swizzled_input()
                else None
            )
            hc = layer.attn_hc if self.attention else layer.ffn_hc
            x = mhc.combine(hc, residual, quantized)
            stats = residual.stats
            residual.release()
            return x, ReadState(
                streams,
                quantized=quantized[0] if quantized else None,
                stats=stats,
                stats_stream=stream,
                call=self.call,
            )
        fn, scale, base = self._parameters()
        x, post, comb, norm_fused = layer.hc_pre(
            residual,
            fn,
            scale,
            base,
            norm=norm if layer.boundary_mhc_fusions else None,
            forward_batch=self.batch,
        )
        quantized = None
        if not norm_fused:
            if self.attention:
                from sglang.srt.models.deepseek_v4 import _is_gfx95_supported

                x, quantized = layer._input_norm(
                    x, allow_aiter_quant=_is_gfx95_supported
                )
            else:
                x = norm(x)
        return x, ReadState(residual, post, comb, quantized, call=self.call)

    def _parameters(self):
        layer = self.layer
        if self.attention:
            return layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
        return layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base

    def update_and_read(self, update, value, residual, norm, **kwargs):
        if not isinstance(update, HcUpdate):
            raise TypeError("mHC read requires its producer's update")
        if self.layer.boundary_pre_from_prev:
            return self.read(
                update.post(value, residual, next=self.next_boundary), norm
            )
        layer = self.layer
        if layer.boundary_mhc_fusions and layer.use_fused_mhc_post_pre:
            from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
                apply_mhc_post_pre_boundary,
            )
            from sglang.srt.models.deepseek_v4 import (
                _MHC_POST_MULT_VALUE,
                _is_gfx95_supported,
            )

            fn, scale, base = self._parameters()
            cached = (
                layer._input_layernorm_weight_bf16
                if self.attention
                else layer._post_attention_layernorm_weight_bf16
            )
            fused = apply_mhc_post_pre_boundary(
                value,
                residual.streams,
                residual.post,
                residual.comb,
                fn,
                scale,
                base,
                layer.hc_mult,
                layer.rms_norm_eps,
                layer.hc_eps,
                _MHC_POST_MULT_VALUE,
                layer.hc_sinkhorn_iters,
                cached if cached is not None else norm.weight.data,
                norm.variance_epsilon,
                fn_transpose=not self.attention,
            )
            if fused is not None:
                streams, x, post, comb, norm_fused = fused
                quantized = None
                if not norm_fused:
                    if self.attention:
                        x, quantized = layer._input_norm(
                            x, allow_aiter_quant=_is_gfx95_supported
                        )
                    else:
                        x = norm(x)
                return x, ReadState(streams, post, comb, quantized, call=self.call)
        return self.read(update.update(value, residual), norm, pending=True)


class MoeFinalize(DeferredFinalize, frozen=True):
    pieces: object = None
    partial: object = None

    def complete(self, *, reduce=True):
        from sglang.srt.layers.moe import post_experts_all_reduce
        from sglang.srt.layers.moe.utils import should_add_replicated_moe_output

        with get_forward().scoped(mlp_reduce_scatter=not reduce):
            value = (
                self.pieces.get_merged() if self.pieces is not None else self.partial
            )
            if reduce:
                value = post_experts_all_reduce(value)
            if (
                self.pieces is not None
                and self.pieces.shared is not None
                and self.pieces.shared_is_replicated
                and should_add_replicated_moe_output()
            ):
                value += self.pieces.shared
            return value


class FinalRead(PendingUpdateRead):
    @staticmethod
    def _complete(output, residual, batch, norm, *, update):
        if (
            not isinstance(output, MoeFinalize)
            or output.pieces is None
            or not update.layer.boundary_pre_from_prev
        ):
            return None
        return update.post(output.pieces, residual)

    completing_fusions = (
        ReadoutFusion(
            SumGroup.MOE_OUTPUT,
            _complete.__func__,
            reads=True,
            carried=True,
            accepts_update=lambda update: isinstance(update, HcUpdate),
        ),
    )

    @staticmethod
    def accepts_pending_update(update):
        return isinstance(update, HcUpdate)

    @staticmethod
    def init_residual(value):
        return value

    @staticmethod
    def read(value, norm):
        return value

    @staticmethod
    def update_and_read(update, value, residual, norm):
        return update.update(value, residual)


def materialize(hidden_states, batch, *, consume=True, next=None):
    """Read mHC streams at a PP, head, Engram or capture seam."""
    stream = residual_batch.stream_of(batch)
    stream.check(hidden_states)
    pending = stream.pending
    if pending is None:
        return hidden_states
    value = pending.for_boundary()
    if isinstance(value, MoeFinalize) and value.pieces is not None:
        result = pending.update.post(value.pieces, stream.residual, next=next)
    elif isinstance(value, UnreducedOutput):
        result = pending.update.post(
            mhc.AttnOutput(value.partial), stream.residual, next=next
        )
    else:
        result = pending.update.post(pending.complete(), stream.residual, next=next)
    if consume:
        stream.write(result)
    return result


def stages(config, layer_id, *, layer=None):
    from sglang.srt.layers.layer_boundary.layout import token_axis_sizes
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    ep = not get_moe_a2a_backend().is_none()
    table_attn, table_ffn, overlapped_ffn = {}, {}, {}
    for variant in (BatchVariant.ORDINARY, BatchVariant.CONTEXT_PARALLEL):
        cp = variant is BatchVariant.CONTEXT_PARALLEL
        source = Layout.sharded_over(
            TokenAxis.ATTN_DP,
            *((TokenAxis.ATTN_CP,) if cp else ()),
            axis_sizes=token_axis_sizes(cp_active=cp),
        )
        compute = Layout.sharded_over(
            *(
                (TokenAxis.ATTN_DP, TokenAxis.ATTN_CP)
                if cp and ep
                else ((TokenAxis.ATTN_DP, TokenAxis.ATTN_TP) if ep else ())
            ),
            axis_sizes=token_axis_sizes(cp_active=cp),
        )
        table_attn[variant] = TransportRows(
            source,
            source,
            source,
            SumGroup.ATTN_TP,
            SumGroup.ATTN_CP if get_parallel().attn_cp_size > 1 else None,
        )
        table_ffn[variant] = TransportRows(
            source, compute, source, None if ep else SumGroup.MOE_OUTPUT
        )
        if not cp:
            overlapped_ffn[variant] = TransportRows(
                source,
                source if ep else compute,
                source,
                None if ep else SumGroup.MOE_OUTPUT,
            )
    if layer is None:
        from sglang.srt.models.deepseek_v4 import (
            _is_fused_mhc_post_pre_enabled_xpu,
            is_cross_layer_mhc_fusion_enabled,
        )

        lagged = config.hc_pre_from_prev_sublayer
        fused = (
            is_cross_layer_mhc_fusion_enabled() or _is_fused_mhc_post_pre_enabled_xpu()
        )
    else:
        lagged = layer.boundary_pre_from_prev
        fused = layer.boundary_mhc_fusions and layer.use_fused_mhc_post_pre
    applied = False if lagged else not fused
    if layer is None:
        read = ReadFacts.of(HcRead(None, True))
        au = UpdateFacts.of(HcUpdate(None, True))
        fu = UpdateFacts.of(HcUpdate(None, False, applied_at_exit=applied))
    else:
        read = HcRead(layer, True)
        au = HcUpdate(layer, True)
        fu = HcUpdate(layer, False, applied_at_exit=applied)
    for variant in (BatchVariant.UNPADDED, BatchVariant.INPUT_SCATTERED):
        table_attn[variant] = table_attn[BatchVariant.ORDINARY]
        table_ffn[variant] = table_ffn[BatchVariant.ORDINARY]
    return (
        declare_attn(
            read=read,
            update=au,
            gathers_attn_tp_input=False,
            row_transport=RowTransport(
                table_attn,
                layer._bind_attention_transport if layer is not None else None,
            ),
        ),
        declare_ffn(
            sparse=True,
            read=HcRead(layer, False) if layer is not None else read,
            update=fu,
            row_transport=RowTransport(
                table_ffn,
                layer._bind_moe_transport if layer is not None else None,
                overlapped_rows=overlapped_ffn,
            ),
        ),
    )


class AttentionTransport(ComputeTransport):
    def __init__(self, layer, hidden_states, batch, call, rows):
        self.rows = rows
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            can_fuse_all_reduce,
        )

        self.input = hidden_states
        self.layer = layer
        self.defers_sum = (
            layer.boundary_mhc_fusions
            and layer.boundary_pre_from_prev
            and layer._can_fuse_attn_mhc
            and can_fuse_all_reduce(hidden_states.shape[0], layer.hc_cfg.hidden)
        )

    @staticmethod
    def scope():
        return nullcontext()

    def prepare_output(self, output):
        if not isinstance(output, mhc.AttnOutput):
            return output
        from sglang.srt.layers.cp.cp_decode_attn_tp import get_cp_decode_attn_tp_ctx

        ctx = get_cp_decode_attn_tp_ctx()
        group = self.rows.decode_group if ctx.use_decode_attn_tp else self.rows.group
        if group is None:
            raise ValueError("attention output owes no declared transport sum")
        return UnreducedOutput(output.partial, group=_sum_group(group))


class MoeTransport(ComputeTransport):
    def __init__(self, layer, hidden_states, forward_batch, call, rows):
        self.rows = rows
        from sglang.srt.layers.cp.interleave import attn_cp_interleave_gather
        from sglang.srt.layers.dp_attention import (
            get_global_dp_buffer,
            is_dp_gatherv_active,
        )
        from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
        from sglang.srt.models.deepseek_v4 import (
            _SHARED_EXPERT_LOCAL,
            _every_row_routed,
            dp_gather_replicate,
            envs,
            get_moe_a2a_backend,
        )

        self.layer = layer
        self.batch = forward_batch
        input_ids, input_ids_global = call.input_ids, call.input_ids_global
        _use_cp = TokenAxis.ATTN_CP in rows.source.sharded
        _use_tp_moe_gather = (
            not _use_cp
            and TokenAxis.ATTN_DP in rows.source.sharded - rows.compute.sharded
        )
        _use_tp_attn_a2a_scatter = (
            not _use_cp
            and TokenAxis.ATTN_TP in rows.compute.sharded - rows.source.sharded
        )
        if (rows.group is None) != (not get_moe_a2a_backend().is_none()):
            raise ValueError("MoE backend does not match its transport output contract")
        # symmetric gather+scatter for the no-EP TP-MoE dp-attn path:
        # all_gatherv gather (in layer.mlp's dp_gather) + reduce_scatterv combine.
        # The experts ARE TP-sharded by intermediate (moe_tp_size==tp_size), so
        # the post-experts reduce is a SUM. reduce_scatterv does that sum+scatter
        # in ONE op, REPLACING the MoE-internal post-experts all_reduce — so we
        # MUST tell the MoE to skip it (mlp_reduce_scatter=True) or it
        # double-reduces. Env-gated via SGLANG_DP_USE_GATHERV, default OFF.
        _use_reduce_scatterv = (
            _use_tp_moe_gather
            and is_dp_gatherv_active()
            and forward_batch.dp_padding_mode is not None
            and not forward_batch.dp_padding_mode.is_max_len()
        )
        # SGLANG_DP_USE_REDUCE_SCATTER: in the MAX_LEN decode path (equal per-rank
        # padding, gatherv inactive, no EP), replace the MoE-internal post-experts
        # all_reduce + dp_scatter with an equal-chunk reduce_scatter. On ROCm this
        # uses the aiter custom kernel (so BOTH gather and combine are aiter custom),
        # elsewhere RCCL reduce_scatter; either way it cuts combine traffic ~2x vs
        # all_reduce. tp_size==attn_dp_size required so the global buffer splits
        # evenly into per-rank chunks.
        _use_reduce_scatter = (
            envs.SGLANG_DP_USE_REDUCE_SCATTER.get()
            and _use_tp_moe_gather
            and not _use_reduce_scatterv
            and not should_use_dp_reduce_scatterv()
            and forward_batch.dp_padding_mode is not None
            and forward_batch.dp_padding_mode.is_max_len()
            and get_parallel().tp_size == get_parallel().attn_dp_size
        )
        mlp_reduce_scatter = (
            _use_cp
            or _use_reduce_scatterv
            or _use_reduce_scatter
            or (_use_tp_moe_gather and should_use_dp_reduce_scatterv())
        )
        # PoC (SGLANG_DP_SHARED_EXPERT_LOCAL): compute the replicated shared expert
        # on LOCAL hidden before the gather and add it back after the combine
        # (reduce_scatterv OR dp_scatter), instead of on the gathered global buffer.
        # Applies to BOTH prefill and decode: the shared expert is a per-token MLP,
        # so computing it on this rank's local tokens (M_local rows) is identical to
        # computing it on the gathered global buffer (M_global rows) and keeping the
        # local slice -- but costs 1/attn_dp_size the rows. With a replicated (TP1) shared
        # expert this cancels the TP1 "full-dim" cost in decode (M_local * dim ==
        # M_global * dim/tp), so decode no longer pays the ~attn_dp_size x penalty.
        _shared_local = None
        _do_shared_local = (
            _SHARED_EXPERT_LOCAL
            and _use_tp_moe_gather
            and getattr(layer.mlp, "shared_experts", None) is not None
            and getattr(layer.mlp, "_shared_expert_tp1", False)
        )
        if _use_cp:
            moe_a2a_backend = get_moe_a2a_backend()
            if moe_a2a_backend.is_none():
                hidden_states = attn_cp_interleave_gather(hidden_states)
            else:
                assert (
                    moe_a2a_backend.is_deepep()
                    or moe_a2a_backend.is_megamoe()
                    or moe_a2a_backend.is_mori()
                ), (
                    "CP requires moe_a2a_backend in ('deepep', 'megamoe', 'mori'), "
                    f"got {moe_a2a_backend.value!r}."
                )
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_parallel().tp_group),
                hidden_states,
            )
            if _do_shared_local and local_hidden_states.shape[0] > 0:
                _shared_local = layer.mlp._forward_shared_experts(local_hidden_states)
            # self_attn has already reduced across attention TP, so these hidden
            # states are replicated and must not be summed by a partial gather.
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        _a2a_scatter_chunks = None
        if _use_tp_attn_a2a_scatter:
            s, r = get_parallel().attn_tp_size, get_parallel().attn_tp_rank
            _a2a_scatter_chunks = list(hidden_states.tensor_split(s))
            hidden_states = _a2a_scatter_chunks[r].contiguous()
            # DSpark next-token layers are not hash-routed and intentionally do not
            # carry token IDs. Only split IDs for callers that actually provide them.
            if input_ids is not None:
                input_ids = input_ids.tensor_split(s)[r].contiguous()
            if input_ids_global is not None:
                input_ids_global = input_ids_global.tensor_split(s)[r].contiguous()
        # Skip the MoE-internal post-experts all_reduce when we will do the
        # reduce via reduce_scatterv/reduce_scatter at the combine below
        # (else double-reduce).
        gathered_rows = (
            _every_row_routed(forward_batch, hidden_states.shape[0])
            if _use_cp and get_moe_a2a_backend().is_none()
            else nullcontext()
        )
        self.input = hidden_states
        self.input_ids = input_ids
        self.input_ids_global = input_ids_global
        self._use_cp = _use_cp
        self._use_tp_moe_gather = _use_tp_moe_gather
        self._use_tp_attn_a2a_scatter = _use_tp_attn_a2a_scatter
        self._use_reduce_scatterv = _use_reduce_scatterv
        self._return_reduce_scatterv = (
            should_use_dp_reduce_scatterv() or _use_reduce_scatterv
        )
        self._use_reduce_scatter = _use_reduce_scatter
        self.mlp_reduce_scatter = mlp_reduce_scatter
        self.sum_in_return = mlp_reduce_scatter and rows.group is not None
        self._shared_local = _shared_local
        self._do_shared_local = _do_shared_local
        self._a2a_scatter_chunks = _a2a_scatter_chunks
        self.gathered_rows = gathered_rows
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            can_fuse_all_reduce,
        )

        self.return_moe_output = self.defers_sum = (
            layer.boundary_mhc_fusions
            and layer.boundary_pre_from_prev
            and layer._can_fuse_ffn_mhc
            and can_fuse_all_reduce(hidden_states.shape[0], layer.hc_cfg.hidden)
            and not _use_cp
            and not _use_tp_moe_gather
            and not _use_tp_attn_a2a_scatter
        )

    @contextmanager
    def scope(self):
        from contextlib import ExitStack

        with ExitStack() as scope:
            scope.enter_context(
                get_forward().scoped(
                    mlp_reduce_scatter=self.mlp_reduce_scatter,
                    defer_moe_finalize=False,
                )
            )
            scope.enter_context(self.gathered_rows)
            yield

    def prepare_output(self, hidden_states):
        from sglang.srt.models.deepseek_v2 import MoEOutput

        if isinstance(hidden_states, MoEOutput):
            completion = MoeFinalize(hidden_states)
            return (
                completion.complete(reduce=False)
                if self.mlp_reduce_scatter
                else completion
            )
        if self.rows.group is not None and not self.mlp_reduce_scatter:
            return MoeFinalize(partial=hidden_states)
        return hidden_states

    def return_output(self, hidden_states):
        if self.defers_sum:
            return hidden_states
        from sglang.srt.layers.dp_attention import (
            attn_tp_all_gather,
            dp_reduce_scatter_tensor,
            dp_scatter,
            get_dp_global_num_tokens,
            get_local_dp_buffer,
        )
        from sglang.srt.layers.layer_boundary.ops import (
            attn_cp_interleave_reduce_scatter,
        )
        from sglang.srt.models.deepseek_v4 import (
            get_moe_a2a_backend,
        )

        forward_batch = self.batch
        _use_cp = self._use_cp
        _use_tp_moe_gather = self._use_tp_moe_gather
        _use_tp_attn_a2a_scatter = self._use_tp_attn_a2a_scatter
        _use_reduce_scatterv = self._use_reduce_scatterv
        _use_reduce_scatter = self._use_reduce_scatter
        _shared_local = self._shared_local
        _a2a_scatter_chunks = self._a2a_scatter_chunks
        if _use_cp and get_moe_a2a_backend().is_none():
            hidden_states = attn_cp_interleave_reduce_scatter(hidden_states)
        elif _use_tp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_parallel().tp_group),
                hidden_states,
            )
            if self._return_reduce_scatterv:
                # SUM the TP-sharded per-rank partial expert outputs AND scatter
                # each rank its own token slice, in one op. Correct because the
                # MoE-internal all_reduce was skipped (mlp_reduce_scatter above).
                # This is the symmetric inverse of the all_gatherv gather.
                get_parallel().tp_group.reduce_scatterv(
                    global_hidden_states,
                    output=hidden_states,
                    sizes=get_dp_global_num_tokens(),
                )
            elif _use_reduce_scatter:
                # Equal-chunk reduce_scatter: SUM the TP-sharded per-rank partial
                # expert outputs AND scatter each rank its own (MAX_LEN-padded)
                # token chunk in one op (symmetric inverse of the MAX_LEN
                # all_gather). Correct because the MoE-internal all_reduce was
                # skipped (mlp_reduce_scatter above). dp_reduce_scatter_tensor
                # routes to the equal-chunk reduce_scatter_tensor here (its
                # variable-length reduce_scatterv branch is gated by
                # is_dp_gatherv_active(), which is False under MAX_LEN), which in
                # turn uses the aiter custom kernel when it fits (else RCCL).
                dp_reduce_scatter_tensor(hidden_states, global_hidden_states)
            else:
                dp_scatter(hidden_states, global_hidden_states, forward_batch)
            # PoC: add the locally-computed shared-expert output to this rank's
            # reduce-scattered / dp-scattered local slice (skipped inside self.mlp
            # above). Covers both prefill (gatherv) and decode (dp_scatter).
            if _shared_local is not None:
                n = hidden_states.shape[0]
                hidden_states = hidden_states + _shared_local[:n]
        if _use_tp_attn_a2a_scatter:
            assert _a2a_scatter_chunks is not None
            gathered = [torch.empty_like(t) for t in _a2a_scatter_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)
        return hidden_states


class OverlappedMoeTransport(ComputeTransport):
    is_overlapped = True

    """The existing TBO issue/wait protocol, owned by one microbatch.

    EP ops already return source rows. The TP path gathers normalized 2-D
    input and sums/scatters expert output asynchronously; residual coefficients
    stay on the source rows throughout. Neither finish path sums again.
    """

    def __init__(self, layer, hidden_states, batch, call, rows):
        self.rows = rows
        from sglang.srt.layers.moe import get_moe_a2a_backend
        from sglang.srt.models.deepseek_v4 import _SHARED_EXPERT_LOCAL

        self.layer, self.input, self.batch = layer, hidden_states, batch
        self.ep = not get_moe_a2a_backend().is_none()
        self.sub = call.tbo_subbatch_index
        self.do_shared_local = (
            not self.ep
            and _SHARED_EXPERT_LOCAL
            and getattr(layer.mlp, "shared_experts", None) is not None
            and getattr(layer.mlp, "_shared_expert_tp1", False)
        )
        self.shared_local = None
        self.gather_event = self.combine_event = None
        self.gather_keepalive = self.combine_keepalive = None
        self.global_hidden = self.local_out = None
        self.combined = False

    def release(self):
        events = (self.gather_event, self.combine_event)
        if any(event is not None for event in events):
            current = torch.cuda.current_stream()
            for event in events:
                if event is not None:
                    current.wait_event(event)
        self.gather_event = self.combine_event = None
        self.gather_keepalive = self.combine_keepalive = None
        self.global_hidden = self.local_out = self.shared_local = None

    def issue_gather(self):
        from sglang.srt.layers.dp_attention import (
            _tbo_event,
            dp_gather_partial,
            get_dp_tbo_comm_stream,
            get_global_dp_buffer_len,
            get_tbo_persistent_buffer,
        )

        if self.ep or self.gather_event is not None or self.global_hidden is not None:
            raise RuntimeError("TBO gather cannot be issued twice")
        local = self.input
        if self.do_shared_local and local.shape[0] > 0:
            self.shared_local = self.layer.mlp._forward_shared_experts(local)
        self.global_hidden = get_tbo_persistent_buffer(
            ("gh", self.sub),
            get_global_dp_buffer_len(),
            local.shape[1],
            local.dtype,
            local.device,
        )
        comm = get_dp_tbo_comm_stream()
        compute = torch.cuda.current_stream()
        with torch.cuda.stream(comm):
            comm.wait_stream(compute)
            dp_gather_partial(self.global_hidden, local, self.batch)
            self.gather_event = _tbo_event(("gather", self.sub))
            self.gather_event.record(comm)
        self.gather_keepalive = local

    def wait_gather(self):
        if self.gather_event is None:
            raise RuntimeError("TBO gather was not issued")
        torch.cuda.current_stream().wait_event(self.gather_event)
        self.gather_event = self.gather_keepalive = None
        return self.global_hidden

    def issue_combine(self, output):
        from sglang.srt.layers.dp_attention import (
            dp_reduce_scatterv_async,
            get_dp_global_num_tokens,
            get_local_dp_buffer_len,
            get_tbo_persistent_buffer,
        )

        if self.ep or self.combine_event is not None or self.combined:
            raise RuntimeError("TBO combine cannot be issued twice")
        self.local_out = get_tbo_persistent_buffer(
            ("lo", self.sub),
            get_local_dp_buffer_len(),
            output.shape[1],
            output.dtype,
            output.device,
        )
        self.combine_event = dp_reduce_scatterv_async(
            self.local_out,
            output,
            get_dp_global_num_tokens(),
            event_key=("combine", self.sub),
        )
        self.combine_keepalive = output
        self.global_hidden = None

    def wait_combine(self):
        if self.combine_event is None:
            raise RuntimeError("TBO combine was not issued")
        torch.cuda.current_stream().wait_event(self.combine_event)
        self.combine_event = self.combine_keepalive = None
        value = self.local_out
        self.local_out = None
        if self.shared_local is not None:
            value = value + self.shared_local[: value.shape[0]]
        self.shared_local = None
        self.combined = True
        return value

    def prepare_output(self, output):
        if not self.ep and not self.combined:
            raise RuntimeError("TBO finish requires the combine wait")
        return output
