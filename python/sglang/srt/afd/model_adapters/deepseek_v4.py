"""DeepSeek V4: A-owned hyper-connections and real Hash routing operands."""

from __future__ import annotations

from typing import Any

from ..config import AFDExecutionMode
from ..contracts import AFDError, AFDRole
from ..metadata import DSV4MetadataGuard
from .base import AFDDecoderAdapter, stage_forward_context


def owns_checkpoint_weight(name: str, mode: AFDExecutionMode) -> bool:
    """Split native checkpoint names before their normal remapping/quantization."""
    ffn = ".ffn." in name or ".mlp." in name
    if mode == AFDExecutionMode.FFN:
        return ffn
    return not ffn if mode == AFDExecutionMode.ATTENTION else True


def matches_deepseek_v4(model: Any) -> bool:
    try:
        config = model.model.config
        return (
            type(model).__name__ == "DeepseekV4ForCausalLM"
            and config.model_type == "deepseek_v4"
            and config.hc_mult > 1
        )
    except AttributeError:
        return False


class DeepseekV4AFDAdapter(AFDDecoderAdapter):
    guard_class = DSV4MetadataGuard
    family_error = "AFD_DEEPSEEK_V4_REQUIRED"
    requires_token_ids = True

    def matches_family(self) -> bool:
        return matches_deepseek_v4(self.model)

    def validate_model(self) -> None:
        super().validate_model()
        config = self.inner.config
        if config.hc_pre_from_prev_sublayer or config.engram_layer_ids:
            raise AFDError("AFD_DSV4_HC_SCHEME_UNSUPPORTED")

    def validate_merge_reduce_scatter(self) -> None:
        from sglang.srt.layers.moe import get_moe_a2a_backend
        from sglang.srt.layers.moe.mega_moe import is_mega_moe_experts_ready

        if not get_moe_a2a_backend().is_megamoe():
            return super().validate_merge_reduce_scatter()
        # MegaMoE owns expert dispatch/combine and returns complete local rows.
        # Its replicated shared expert must not enter the TP partial-sum path.
        for index, layer in enumerate(self.inner.layers):
            if not is_mega_moe_experts_ready(layer.mlp.experts):
                raise AFDError("AFD_DSV4_MEGAMOE_WEIGHTS_NOT_READY", f"layer={index}")

    def _group_ffn_output(self, *, decoder_layer, stage, lane_hidden_states):
        import torch

        from sglang.srt.environ import envs
        from sglang.srt.layers.moe import get_moe_a2a_backend
        from sglang.srt.layers.moe.mega_moe import (
            forward_mega_moe,
            is_mega_moe_experts_ready,
        )

        if not get_moe_a2a_backend().is_megamoe():
            return super()._group_ffn_output(
                decoder_layer=decoder_layer,
                stage=stage,
                lane_hidden_states=lane_hidden_states,
            )
        if not is_mega_moe_experts_ready(decoder_layer.mlp.experts):
            raise AFDError("AFD_DSV4_MEGAMOE_WEIGHTS_NOT_READY")
        rows = tuple(value.shape[0] for value in lane_hidden_states)
        widths = stage.group_widths
        if len(rows) != len(widths) or any(r > w for r, w in zip(rows, widths)):
            raise AFDError("AFD_FFN_MERGE_WIDTH_EXCEEDED")
        strides = widths if len(stage.merge_sizes) > 1 else rows
        total = sum(strides)
        # An F rank without ingress still participates in expert collectives.
        # One discarded zero-input row produces a real kernel output for the
        # role-graph replay sentinel; it is never returned to an A request.
        capacity = max(total, 1)
        if capacity > envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get():
            raise AFDError("AFD_DSV4_MEGAMOE_LOCAL_ROWS_EXCEED_CAPACITY")
        template = lane_hidden_states[0] if rows else stage.collective_input
        if template is None or (not rows and template.shape[0] != 0):
            raise AFDError("AFD_FFN_EMPTY_INGRESS_TEMPLATE_INVALID")
        hidden = template.new_zeros((capacity, *template.shape[1:]))
        offset = 0
        for value, row, stride in zip(lane_hidden_states, rows, strides):
            hidden[offset : offset + row].copy_(value)
            offset += stride
        ids = None
        if decoder_layer.mlp.is_hash:
            if stage.merged_input_ids is None:
                if stage.input_ids is None or len(stage.input_ids) != len(rows):
                    raise AFDError("AFD_DSV4_HASH_INPUT_IDS_REQUIRED")
                id_template = stage.input_ids[0] if rows else stage.collective_input_ids
                if id_template is None or id_template.dtype != torch.int64:
                    raise AFDError("AFD_DSV4_HASH_INPUT_IDS_DTYPE_INVALID")
                ids = id_template.new_zeros((capacity,))
                offset = 0
                for value, row, stride in zip(stage.input_ids, rows, strides):
                    if value.dtype != torch.int64 or value.numel() < row:
                        raise AFDError("AFD_DSV4_HASH_INPUT_IDS_ROWS_INVALID")
                    ids[offset : offset + row].copy_(value.reshape(-1)[:row])
                    offset += stride
                stage.merged_input_ids = ids
            ids = stage.merged_input_ids
            if ids.shape != (capacity,):
                raise AFDError("AFD_DSV4_HASH_INPUT_IDS_ROWS_INVALID")
        # Padded lane blocks are not one real-token prefix. Keep routing's
        # non-prefix context and let native MegaMoE own scales, clamp and shared
        # experts. No extra all-gather, reduction or routed-factor application.
        with stage_forward_context(None):
            output = forward_mega_moe(
                decoder_layer.mlp,
                hidden,
                stage.forward_batch,
                input_ids_global=ids,
            )
        if output.shape != hidden.shape:
            raise AFDError("AFD_DSV4_MEGAMOE_OUTPUT_SHAPE_INVALID")
        if not rows:
            stage.graph_output = output
        result = []
        offset = 0
        for row, stride in zip(rows, strides):
            result.append(output.narrow(0, offset, row))
            offset += stride
        return tuple(result)

    def split_step(self, **kwargs):
        stages = super().split_step(**kwargs)
        for stage in stages:
            # Compressor phase caches describe one invocation's token rows.
            for name in ("freqs_cis_c4", "freqs_cis_c128"):
                stage.forward_batch.__dict__.pop(name, None)
        return stages

    def prepare_stage(self, *, stage):
        super().prepare_stage(stage=stage)
        if self.role == AFDRole.ATTENTION:
            self.attention_backend._activate_request_window()

    def local_compute(self, *, layer, stage, hidden_states, residual, positions=None):
        if self.role == AFDRole.FFN:
            return super().local_compute(
                layer=layer,
                stage=stage,
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
            )
        if hidden_states.ndim != 3:
            raise AFDError("AFD_DSV4_MHC_STREAM_SHAPE_INVALID")
        return (
            self.inner.layers[layer].forward_attention_for_afd(
                positions=stage.positions if positions is None else positions,
                hidden_states=hidden_states,
                forward_batch=stage.forward_batch,
            ),
            None,
        )

    def finish_layer(self, *, layer, stage, ffn_output, residual):
        if self.role != AFDRole.ATTENTION:
            return ffn_output, residual
        from sglang.srt.layers.layer_boundary.residual import batch
        from sglang.srt.models.deepseek_v4_boundary import FinalRead

        forward_batch = stage.forward_batch
        boundary = self.inner.layers[layer].ffn_boundary
        try:
            with boundary.exit(forward_batch) as output:
                hidden = output.finish(ffn_output)
            # Materialize using the native mHC update, never PLAIN_ADD. Keeping
            # the inter-layer seam tensor-only also makes role capture stable.
            streams = batch.read_output(hidden, forward_batch, FinalRead(), None)
        finally:
            batch.clear(forward_batch)
        return streams, None

    def make_ffn_stages(self, *, descriptor, buffers, lane, shape):
        if len(buffers) % 2:
            raise AFDError("AFD_DSV4_INPUT_ID_BACKING_MISSING")
        count = len(buffers) // 2
        stages, views = super().make_ffn_stages(
            descriptor=descriptor, buffers=buffers[:count], lane=lane, shape=shape
        )
        group = len(shape.group_lanes(ffn_ordinal=lane))
        slots = max(1, group)
        for index, stage in enumerate(stages):
            ids = buffers[count + index * slots : count + (index + 1) * slots]
            stage.input_ids = ids[:group]
            stage.collective_input_ids = ids[0] if group == 0 else None
        return stages, views

    def _compute_ffn_output(
        self, decoder_layer, hidden_states, stage, *, rows, strides
    ):
        import torch

        from sglang.srt.runtime_context import get_parallel

        ids = None
        if decoder_layer.mlp.is_hash:
            if stage.merged_input_ids is None:
                if stage.input_ids is None or len(stage.input_ids) != len(rows):
                    raise AFDError("AFD_DSV4_HASH_INPUT_IDS_REQUIRED")
                template = (
                    stage.input_ids[0]
                    if stage.input_ids
                    else stage.collective_input_ids
                )
                if template is None or template.dtype != torch.int64:
                    raise AFDError("AFD_DSV4_HASH_INPUT_IDS_DTYPE_INVALID")
                local = template.new_zeros((sum(strides), 1))
                offset = 0
                for value, row, stride in zip(stage.input_ids, rows, strides):
                    local[offset : offset + row].copy_(value[:row])
                    offset += stride
                if len(stage.merge_sizes) > 1:
                    local = get_parallel().tp_group.all_gatherv(
                        local, sizes=list(stage.merge_sizes)
                    )[0]
                stage.merged_input_ids = local.reshape(-1)
            ids = stage.merged_input_ids
            if ids.shape[0] != hidden_states.shape[0]:
                raise AFDError("AFD_DSV4_HASH_INPUT_IDS_ROWS_INVALID")
        return decoder_layer.compute_ffn_output(
            hidden_states, stage.forward_batch, input_ids=ids
        )
