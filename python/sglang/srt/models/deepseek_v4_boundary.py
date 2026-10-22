"""DeepSeek hyper-connection reads, updates and operand row transport."""

from functools import partial

import msgspec
import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import get_cp_decode_attn_tp_ctx
from sglang.srt.layers.layer_boundary.contracts import BatchVariant, ReadoutFusion
from sglang.srt.layers.layer_boundary.factories import declare_attn, declare_ffn
from sglang.srt.layers.layer_boundary.facts import ReadFacts, UpdateFacts
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    _sum_group,
)
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import PendingUpdateRead
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.row_transport import (
    ComputeTransport,
    RowTransport,
    TransportRows,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    can_fuse_all_reduce,
)
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common.moe_transport import (
    MoeFinalize,
    MoeRowTransport,
)
from sglang.srt.models.deepseek_v2 import MoEOutput
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_gfx95_supported

_is_gfx95_supported = is_gfx95_supported()


class LayerCall(msgspec.Struct, frozen=True):
    input_ids: object = None
    input_ids_global: object = None
    allow_cross_layer_fusion: bool = True
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
        if self.attention:
            return mhc.run_attn_post(
                hc,
                output,
                state.streams,
                stats_stream=state.stats_stream,
                next=next,
                world_size=self.layer.self_attn.attn_tp_size,
                precomputed=state.stats,
            )
        return mhc.run_moe_post(
            hc,
            output,
            state.streams,
            stats_stream=state.stats_stream,
            next=next if state.call.allow_cross_layer_fusion else None,
            world_size=self.layer.mlp.tp_size,
            precomputed=state.stats,
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
            accepts_mxfp8_deepgemm=self.attention
            and self.layer.self_attn.accepts_deepgemm_mxfp8_input(),
        )

    def _after_post(self, residual, norm):
        return self.read(residual, norm)

    def read(
        self,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
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
                mhc.MHC_POST_MULT_VALUE,
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
        return self.read(update.update(value, residual), norm)


class FinalRead(PendingUpdateRead):
    bind = None

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
    # TBO keeps source rows until expert dispatch, including an uneven final
    # microbatch. It does not shard these rows across attention TP.
    overlapped_ffn[BatchVariant.UNPADDED] = overlapped_ffn[BatchVariant.ORDINARY]
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
        if rows.source != rows.compute or rows.source != rows.returned:
            raise ValueError("attention transport must preserve its source rows")
        if rows.group is not SumGroup.ATTN_TP or rows.decode_group not in (
            None,
            SumGroup.ATTN_CP,
        ):
            raise ValueError("attention transport does not match its projection groups")
        self.num_rows = hidden_states.shape[0]
        self.rows = rows
        self.input = hidden_states
        self.defers_sum = (
            layer.boundary_mhc_fusions
            and layer.boundary_pre_from_prev
            and layer._can_fuse_attn_mhc
            and can_fuse_all_reduce(hidden_states.shape[0], layer.hc_cfg.hidden)
        )

    def release(self):
        self.input = None

    def prepare_output(self, output):
        self.input = None
        value = output.partial if isinstance(output, mhc.AttnOutput) else output
        if value.shape[0] != self.num_rows:
            raise ValueError("attention output changed its source rows")
        if not isinstance(output, mhc.AttnOutput):
            return output
        ctx = get_cp_decode_attn_tp_ctx()
        group = self.rows.decode_group if ctx.use_decode_attn_tp else self.rows.group
        if group is None:
            raise ValueError("attention output owes no declared transport sum")
        return UnreducedOutput(output.partial, group=_sum_group(group))


class MoeTransport(MoeRowTransport):
    def __init__(self, layer, hidden_states, forward_batch, call, rows):
        self.source_rows = hidden_states.shape[0]
        self.rows = rows
        if (rows.group is None) != (not get_moe_a2a_backend().is_none()):
            raise ValueError("MoE backend does not match its transport output contract")
        cp = TokenAxis.ATTN_CP in rows.source.sharded
        super().__init__(
            layer.mlp,
            hidden_states,
            forward_batch,
            call.input_ids,
            call.input_ids_global,
            cp_return_contiguous=layer.config.model_type == "deepseek_v41",
            use_cp=cp,
            moe_gather=not cp
            and TokenAxis.ATTN_DP in rows.source.sharded - rows.compute.sharded,
            attn_scatter=not cp
            and TokenAxis.ATTN_TP in rows.compute.sharded - rows.source.sharded,
        )
        self.sum_in_return = self.mlp_reduce_scatter and rows.group is not None
        self.return_moe_output = self.defers_sum = (
            layer.boundary_mhc_fusions
            and layer.boundary_pre_from_prev
            and layer._can_fuse_ffn_mhc
            and can_fuse_all_reduce(self.input.shape[0], layer.hc_cfg.hidden)
            and not self._use_cp
            and not self._use_tp_moe_gather
            and not self._use_tp_attn_a2a_scatter
        )

    def prepare_output(self, hidden_states):
        self.input = None
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
        value = super().return_output(hidden_states)
        if value.shape[0] != self.source_rows:
            raise ValueError("MoE transport did not return its source rows")
        return value
