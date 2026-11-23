"""Compute-owned GLM MoE row transport, selected by the layer boundary.

Route BF16 shards locally, then gather FP8 values, scales and exact routing
results through the same CP helper as the BF16 fallback. Residuals stay on
source rows; the invocation returns the reduced output to those rows.
"""

from contextlib import contextmanager, nullcontext
from functools import cached_property

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.interleave import attn_cp_interleave_gather
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    _sum_group,
    token_axis_sizes,
)
from sglang.srt.layers.layer_boundary.ops import attn_cp_interleave_reduce_scatter
from sglang.srt.layers.layer_boundary.output import UnreducedOutput
from sglang.srt.layers.layer_boundary.row_transport import (
    ComputeTransport,
    RowTransport,
    TransportRows,
)
from sglang.srt.runtime_context import get_forward, get_parallel


def resolve_cp_moe_fp8_transport() -> bool:
    """Resolve the opt-in and topology once, before declaring a model's layers."""
    if not envs.SGLANG_CP_MOE_FP8_TRANSPORT.get():
        return False
    from sglang.srt.layers import layernorm_sp
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend

    p = get_parallel()
    runner = get_moe_runner_backend()
    return (
        p.enable_prefill_cp
        and p.enable_cp_tp_group_sharing
        and p.cp_strategy == "interleave"
        and p.attn_cp_size == p.tp_size
        and p.attn_cp_size > 1
        and p.attn_tp_size == 1
        and p.attn_dp_size == 1
        and p.moe_dp_size == 1
        and p.moe_ep_size == 1
        and p.pp_size == 1
        and not p.enable_cp_decode_attn_tp
        and not p.enable_attn_tp_input_scattered
        and not layernorm_sp.layernorm_sp_enabled()
        and get_moe_a2a_backend().is_none()
        and (runner.is_triton() or runner.is_deep_gemm())
    )


def cp_moe_row_transport(bind=None) -> RowTransport:
    """Declare rows without re-evaluating the opt-in or selecting a batch.

    Only the boundary selects the variant. Non-CP invocations keep BF16 MoE;
    a CP invocation owns exactly one FP8 or BF16 gather.
    """
    rows = {}
    for variant in (
        BatchVariant.ORDINARY,
        BatchVariant.CONTEXT_PARALLEL,
        BatchVariant.UNPADDED,
    ):
        axes = token_axis_sizes(cp_active=variant is BatchVariant.CONTEXT_PARALLEL)
        source = Layout.sharded_over(
            TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axes
        )
        compute = Layout.sharded_over(axis_sizes=axes)
        rows[variant] = TransportRows(source, compute, source, SumGroup.MOE_OUTPUT)
    return RowTransport(rows, bind)


class CpMoeFp8Binding:
    """Per-layer binding; no layer-id registry or process-global mutable state."""

    def __init__(self, layer):
        self.layer = layer

    @cached_property
    def eligibility(self):
        # Runner/quantization state is finalized after weight loading.
        return self.layer._compute_cp_moe_fp8_transport_enabled()

    def __call__(self, hidden_states, forward_batch, context, rows, overlapped):
        return CpMoeFp8Invocation(self, hidden_states, forward_batch, rows)


class CpMoeFp8Invocation(ComputeTransport):
    def __init__(self, binding, hidden_states, forward_batch, rows):
        self.binding = binding
        self.rows = rows
        self.batch = forward_batch
        self.source_rows = hidden_states.shape[0]
        self.is_cp = TokenAxis.ATTN_CP in rows.source.sharded - rows.compute.sharded
        self.sum_in_return = self.is_cp
        self.prepared = None
        if self.is_cp and self._can_quantize(hidden_states):
            self.prepared = self._prepare_fp8(hidden_states)
            self.input = hidden_states
        else:
            self.input = (
                attn_cp_interleave_gather(hidden_states)
                if self.is_cp
                else hidden_states
            )

    def _can_quantize(self, hidden_states):
        if (
            not self.binding.eligibility[0]
            or not hidden_states.is_cuda
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or hidden_states.shape[0] < 4
            or hidden_states.shape[0] % 4
        ):
            return False
        layer = self.binding.layer
        if layer.experts.quant_method.runner.runner_backend.is_deep_gemm():
            # Compact prequantized input does not support masked-standard
            # layout. Preserve the baseline GEMM choice for those shapes.
            from sglang.srt.layers.moe.moe_runner.deep_gemm import (
                DeepGemmMoeQuantInfo,
                _should_use_masked_standard_layout,
                deep_gemm_sm120,
            )

            experts = layer.experts
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=experts.w13_weight,
                w2_weight=experts.w2_weight,
                use_fp8=True,
                w13_scale=experts.w13_weight_scale_inv,
                w2_scale=experts.w2_weight_scale_inv,
                block_shape=experts.quant_method.quant_config.weight_block_size,
            )
            # This view is metadata only. The policy reads shape/dtype/device,
            # not values; no full-size BF16 allocation or gather is needed.
            full_shape = hidden_states.as_strided(
                (
                    self.source_rows * get_parallel().attn_cp_size,
                    hidden_states.shape[1],
                ),
                (0, 1),
            )
            if (
                deep_gemm_sm120.allows_masked_standard_layout()
                and _should_use_masked_standard_layout(
                    layer.experts.moe_runner_config,
                    quant_info,
                    full_shape,
                )
            ):
                return False
        return True

    def _prepare_fp8(self, hidden_states):
        from sglang.kernels.ops.quantization.moe_fp8_packet import (
            pack_moe_fp8,
            unpack_moe_fp8,
        )
        from sglang.kernels.ops.quantization.per_token_group_quant import (
            per_token_group_quant,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        layer = self.binding.layer
        topk = layer.topk(
            hidden_states, layer.gate(hidden_states), num_token_non_padded=None
        )
        assert TopKOutputChecker.format_is_standard(topk)
        # Route original BF16 rows, never a lossy FP8 reconstruction.
        if layer.experts.quant_method.runner.runner_backend.is_deep_gemm():
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8_row_padded,
            )

            q, scale = sglang_per_token_group_quant_fp8_row_padded(hidden_states, 128)
        else:
            q, scale = per_token_group_quant(hidden_states, group_size=128)
        payload = pack_moe_fp8(q, scale, topk.topk_weights, topk.topk_ids)
        gathered = attn_cp_interleave_gather(payload)
        return unpack_moe_fp8(
            gathered,
            hidden_states.shape[1],
            topk.topk_ids.shape[1],
            column_major_scales=False,
        )

    @contextmanager
    def scope(self):
        from sglang.srt.models.deepseek_common.moe_transport import _every_row_routed

        with (
            get_forward().scoped(
                mlp_reduce_scatter=self.is_cp, defer_moe_finalize=False
            ),
            _every_row_routed(self.batch, self.input.shape[0])
            if self.is_cp and self.prepared is None
            else nullcontext(),
        ):
            yield

    def forward(self, forward_batch, zero_allocator=None):
        layer = self.binding.layer
        if self.prepared is None:
            return layer(self.input, forward_batch, zero_allocator)
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            maybe_fuse_routed_scale_and_shared_add,
        )

        q, scale, weights, ids = self.prepared
        # Both consumers read pre_quant_input; this is shape/dtype/output
        # metadata only, never an initialized BF16 GEMM operand.
        shape_only = self.input.new_empty(q.shape)
        # Separate shared linears require TMA-aligned column-major scales.
        # Only those consumers pay a conversion; routed compact DeepGEMM
        # consumes the row-major scales emitted directly by unpack.
        shared_scale = (
            scale.t().contiguous().t() if layer.shared_experts is not None else scale
        )
        shared = layer._forward_shared_experts(
            shape_only, pre_quant_input=(q, shared_scale)
        )
        routed = layer.experts.forward_impl(
            shape_only,
            StandardTopKOutput(weights, ids, None),
            pre_quant_input=(q, scale),
        )
        output = maybe_fuse_routed_scale_and_shared_add(
            layer.experts,
            routed,
            None if layer._shared_expert_tp1 else shared,
            layer.routed_scaling_factor,
        )
        if (
            shared is not None
            and layer._shared_expert_tp1
            and layer._adds_replicated_output()
        ):
            output += shared
        return output

    def prepare_output(self, output):
        if self.sum_in_return:
            return output
        return UnreducedOutput(output, group=_sum_group(self.rows.group))

    def return_output(self, output):
        if self.is_cp:
            output = attn_cp_interleave_reduce_scatter(output)
        if output.shape[0] != self.source_rows:
            raise ValueError("CP MoE transport did not return its source rows")
        return output

    def release(self):
        self.input = self.prepared = None
