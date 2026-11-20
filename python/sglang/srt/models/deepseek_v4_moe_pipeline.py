"""Opt-in row-wise MoE and epilogue pipelines for CUDA context-parallel prefill.

This module owns invocation-local buffers. The only shared objects are CUDA
communication streams, cached by device and priority. Attention, its KV stores,
and its cross-token dependencies remain in the ordinary model path.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RowPipelineConfig:
    microbatches: int = 0
    min_rows: int = 1024
    packed_gather: bool = False
    epilogue: bool = False
    query_lookahead: bool = False
    comm_priority: int = 0

    def __post_init__(self):
        if self.microbatches < 0:
            raise ValueError("MoE microbatches must be nonnegative (0 disables)")
        if self.min_rows < 1:
            raise ValueError("MoE min_rows must be positive")
        if self.comm_priority > 0:
            raise ValueError("CUDA stream priority must be zero or negative")

    @classmethod
    def from_env(cls):
        from sglang.srt.environ import envs

        return cls(
            envs.SGLANG_DSV4_MOE_MICROBATCHES.get(),
            envs.SGLANG_DSV4_MOE_MIN_ROWS.get(),
            envs.SGLANG_DSV4_MOE_PACKED_GATHER.get(),
            envs.SGLANG_DSV4_MOE_EPILOGUE_PIPELINE.get(),
            envs.SGLANG_DSV4_QUERY_LOOKAHEAD.get(),
            envs.SGLANG_DSV4_MOE_COMM_PRIORITY.get(),
        )


def row_bounds(num_rows: int, microbatches: int):
    """Cover every local row once; avoid creating sub-256-row tiny GEMMs."""
    if num_rows < 0 or microbatches < 1:
        raise ValueError("rows must be nonnegative and microbatches positive")
    if not num_rows:
        return ()
    count = min(microbatches, max(1, num_rows // 256))
    step = (num_rows + count - 1) // count
    return tuple((a, min(a + step, num_rows)) for a in range(0, num_rows, step))


def submit_row_pipeline(
    num_microbatches, *, prepare, gather, compute, scatter, shared, finish
):
    """Submit work in a rank-consistent order, then drain completed rows.

    The callbacks own stream waits. In particular, finish must wait for its RS
    before closing the residual and invoking any next-layer query lookahead.
    Keeping drain after all compute/shared submissions avoids reordering work
    on the main CUDA stream. AG and RS use the same communication stream.
    """
    if num_microbatches < 1:
        raise ValueError("a row pipeline needs at least one microbatch")
    prepare(0)
    gather(0)
    if num_microbatches > 1:
        prepare(1)
        gather(1)
    for i in range(num_microbatches):
        result = compute(i)
        scatter(i, result)
        if i + 2 < num_microbatches:
            prepare(i + 2)
            gather(i + 2)
        shared(i)
    for i in range(num_microbatches):
        finish(i)


@dataclass
class QueryLookahead:
    batch: Any
    owner: Any
    positions: Any
    input_rows: Any
    qkv_a: Any
    q_lora: Any
    q: Any
    read_state: Any

    def matches(self, batch, owner, positions):
        return (
            self.batch is batch and self.owner is owner and self.positions is positions
        )


_COMM_STREAMS = {}


def _comm_stream(device, priority):
    import torch

    key = (device.index, priority)
    if key not in _COMM_STREAMS:
        with torch.cuda.device(device):
            _COMM_STREAMS[key] = torch.cuda.Stream(device=device, priority=priority)
    return _COMM_STREAMS[key]


def swizzle_scale_factors(sf):
    """Linear UE8M0 scales to FlashInfer's zero-padded 128 by 4 layout."""
    import torch

    rows, cols = sf.shape
    padded_rows, padded_cols = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), dtype=sf.dtype, device=sf.device
        )
        padded[:rows, :cols] = sf
        sf = padded
    return (
        sf.view(padded_rows // 128, 4, 32, padded_cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(-1)
    )


def eligible(layer, rows, batch, config):
    """Conservative gate: unsupported backends retain the normal boundaries."""
    if config.microbatches <= 1 or rows.shape[0] < config.min_rows:
        return False
    import torch

    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.layers.moe.utils import (
        should_use_flashinfer_cutlass_moe_fp4_allgather,
    )
    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
        Mxfp4FlashinferCutlassMoEMethod,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.runtime_context import (
        get_exec,
        get_forward,
        get_parallel,
        get_platform,
    )

    parallel, moe = get_parallel(), layer.mlp
    if not (
        rows.is_cuda
        and get_platform().is_sm120
        and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        and batch.forward_mode.is_extend_without_speculative()
        and layer.dsa_enable_prefill_cp
        and dsa_use_prefill_cp(batch)
        and parallel.attn_cp_size > 1
        and parallel.attn_dp_size == 1
        and parallel.attn_tp_size == 1
        and parallel.moe_tp_size == 1
        and parallel.moe_dp_size == 1
        and parallel.moe_ep_size == parallel.attn_cp_size
        and not get_forward().sp_active
        and get_moe_a2a_backend().is_none()
        and not torch.cuda.is_current_stream_capturing()
        and not is_in_breakable_cuda_graph()
        and layer.hc_stats_stream is None
        and not getattr(moe, "is_hash", False)
        and getattr(moe.topk, "waterfill_balancer", None) is None
        and getattr(moe.gate, "e_score_correction_bias_vl", None) is None
        and getattr(moe, "num_fused_shared_experts", 0) == 0
        and (moe.shared_experts is None or moe._shared_expert_tp1)
    ):
        return False
    options = get_exec().moe
    if (
        options.enable_eplb
        or options.init_expert_location != "trivial"
        or options.ep_num_redundant_experts
        or options.expert_distribution_recorder_mode is not None
    ):
        return False
    method = getattr(moe.experts, "quant_method", None)
    if not (
        isinstance(method, Mxfp4FlashinferCutlassMoEMethod)
        and method._use_mxfp8_act_scaling
        and method._mxfp4_weight_global_scale_tensor is not None
    ):
        return False
    if config.packed_gather and moe.shared_experts is not None:
        linear = moe.shared_experts.gate_up_proj
        quant = linear.quant_method
        backend = getattr(quant, "mxfp8_dense_backend", None)
        if not (
            (
                getattr(quant, "use_mxfp8", False)
                or (
                    getattr(quant, "block_fp8_as_mxfp8", False)
                    and getattr(linear, "block_fp8_mxfp8_ready", False)
                )
            )
            and backend is not None
            and (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
        ):
            return False
    if config.epilogue:
        attention = layer.self_attn
        from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
            is_unified_kv_triton,
        )
        from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

        if not (
            not is_unified_kv_triton()
            and config.packed_gather
            and not getattr(attention, "use_flashinfer_mxfp8_wo_b", False)
            and not attention.use_npu_arch35_mxfp8_wo_a
            and not attention.wo_a_fp8
            and not attention.use_mega_attention
            and getattr(attention.wo_a, "weight", None) is not None
            and getattr(attention.wo_b.quant_method, "mxfp8_dense_backend", None)
            != Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
        ):
            return False
    return True


def can_look_ahead(next_layer, batch, num_rows):
    if next_layer is None or next_layer.engram is not None:
        return False
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton

    attention = next_layer.self_attn
    return (
        next_layer.hc_stats_stream is None
        and not attention.use_fused_qk_norm_rope
        and attention.fuse_wqa_wkv
        and not attention.q_head_norm
        and not attention.use_mega_attention
        and not getattr(attention, "mega_wq_b_permuted", False)
        and attention._kernel_num_heads(num_rows) == attention.n_local_heads
        and not attention.accepts_mxfp8_swizzled_input()
        and not attention.accepts_deepgemm_mxfp8_input()
        and not is_unified_kv_triton()
    )


def take_prepared_attention(layer, owner, batch, positions, call):
    """Consume an invocation-owned cache once, before the normal prepare path."""
    cached = getattr(layer, "_query_lookahead", None)
    layer._query_lookahead = None
    if cached is None or not cached.matches(batch, owner, positions):
        return None
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.models import deepseek_v4_boundary as boundary

    stream = residual_batch.stream_of(batch)
    stream.check(owner)
    if stream.pending is not None or stream.transport is not None:
        return None
    cached.read_state.call = call
    # HcState owns consumed row buffers; a plain Tensor has no release hook.
    release = getattr(owner, "release", None)
    if release is not None:
        release()
    stream.write(cached.read_state)
    variant = layer.attn_boundary.plan.variant_for(batch)
    rows = layer.attn_boundary.declaration.row_transport.rows[variant]
    transport = boundary.AttentionTransport(layer, cached.input_rows, batch, call, rows)
    stream.transport = (layer.attn_boundary, transport)
    layer.self_attn._query_lookahead = cached
    return cached.input_rows


def run_pipeline(layer, attention_output, batch, positions, config, next_layer=None):
    """Complete attention/FFN rows and publish one ordinary written residual."""
    import torch

    from sglang.srt.layers.dp_attention import (
        attn_cp_all_gather_into_tensor,
        attn_cp_reduce_scatter_tensor,
    )
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    from sglang.srt.models import deepseek_v4_boundary as boundary
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.models.deepseek_common.moe_transport import _every_row_routed
    from sglang.srt.runtime_context import get_forward, get_parallel

    state = residual_batch.written_residual(batch)
    stream = residual_batch.stream_of(batch)
    # The full attention input transport has no outstanding communication here
    # (the gate requires attention TP1). The row pipeline owns the output now.
    assert stream.transport[0] is layer.attn_boundary
    stream.transport[1].release()
    stream.transport = None
    rows = attention_output.shape[0]
    bounds = row_bounds(rows, config.microbatches)
    count = len(bounds)
    main = torch.cuda.current_stream()
    comm = _comm_stream(attention_output.device, config.comm_priority)
    source, metadata, gathered = ([None] * count for _ in range(3))
    source_ready, gather_done, compute_done, rs_done = (
        [None] * count for _ in range(4)
    )
    ffn_state, local_input, local_quant, shared_output = (
        [None] * count for _ in range(4)
    )
    results, pre_weights, next_states = [], [], []
    lookahead = config.query_lookahead and can_look_ahead(next_layer, batch, rows)
    query_parts, x_parts = [], []
    output = state.streams.new_empty((rows, layer.hidden_size))
    output.record_stream(comm)
    call = state.call

    def event(stream):
        result = torch.cuda.Event()
        result.record(stream)
        return result

    def slice_state(value, a, b):
        def take(x):
            return None if x is None else x[a:b]

        return boundary.ReadState(
            value.streams[a:b],
            take(value.post),
            take(value.comb),
            stats=None if value.stats is None else tuple(t[a:b] for t in value.stats),
            call=value.call,
        )

    def pack(x):
        from flashinfer import mxfp8_quantize

        from sglang.srt.layers.moe.topk import TopKOutputChecker

        logits = layer.mlp.gate(x, None)
        topk = layer.mlp.topk(x, logits)
        if not TopKOutputChecker.format_is_standard(topk):
            raise RuntimeError("row-wise MoE needs standard top-k output")
        q, scales = mxfp8_quantize(x, is_sf_swizzled_layout=False, alignment=32)
        fields = (q, scales, topk.topk_weights.contiguous(), topk.topk_ids.contiguous())
        sizes = tuple(
            t.reshape(x.shape[0], -1).view(torch.uint8).shape[1] for t in fields
        )
        packed = torch.cat(
            [t.reshape(x.shape[0], -1).view(torch.uint8) for t in fields], dim=1
        )
        return packed, (sizes, tuple(t.dtype for t in fields)), (q, scales)

    def prepare(i):
        a, b = bounds[i]
        projected = attention_output[a:b]
        if config.epilogue:
            projected = layer.self_attn.project_row_epilogue(projected)
        updated = boundary.HcUpdate(layer, True).post(
            projected, slice_state(state, a, b)
        )
        read = boundary.HcRead(layer, False).bind(batch, call)
        x, ffn_state[i] = read.read(updated, layer.post_attention_layernorm)
        local_input[i] = x
        if config.packed_gather:
            source[i], metadata[i], local_quant[i] = pack(x)
        else:
            source[i] = x.contiguous()
        source_ready[i] = event(main)

    def gather(i):
        value = source[i]
        result = value.new_empty(
            (get_parallel().attn_cp_size * value.shape[0], value.shape[1])
        )
        result.record_stream(comm)
        value.record_stream(comm)
        with torch.cuda.stream(comm):
            comm.wait_event(source_ready[i])
            attn_cp_all_gather_into_tensor(result, value)
            gather_done[i] = event(comm)
        gathered[i] = result

    def compute(i):
        main.wait_event(gather_done[i])
        value, gathered[i] = gathered[i], None
        with (
            get_forward().scoped(mlp_reduce_scatter=True),
            _every_row_routed(batch, value.shape[0]),
        ):
            if config.packed_gather:
                from sglang.srt.layers.moe.topk import StandardTopKOutput
                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    maybe_fuse_routed_scale_and_shared_add,
                )

                sizes, dtypes = metadata[i]
                fields = [
                    t.contiguous().view(dtype)
                    for t, dtype in zip(value.split(sizes, dim=1), dtypes)
                ]
                q, scales, weights, ids = fields
                topk = StandardTopKOutput(
                    topk_weights=weights, topk_ids=ids, router_logits=None
                )
                result = layer.mlp.experts(
                    q, topk, pre_quant_input=(q, swizzle_scale_factors(scales), None)
                )
                result = maybe_fuse_routed_scale_and_shared_add(
                    layer.mlp.experts, result, None, layer.mlp.routed_scaling_factor
                )
            else:
                result = layer.mlp(
                    value,
                    batch,
                    input_ids=None,
                    input_ids_global=None,
                    skip_shared_experts=layer.mlp.shared_experts is not None,
                )
        compute_done[i] = event(main)
        return result

    def scatter(i, value):
        a, b = bounds[i]
        value.record_stream(comm)
        with torch.cuda.stream(comm):
            comm.wait_event(compute_done[i])
            attn_cp_reduce_scatter_tensor(output[a:b], value)
            rs_done[i] = event(comm)
        source[i] = None

    def shared(i):
        if layer.mlp.shared_experts is not None:
            if config.packed_gather:
                q, scales = local_quant[i]
                shared_output[i] = layer.mlp._forward_shared_experts(
                    q,
                    None,
                    pre_quant_input=Mxfp8SwizzledInput(
                        q, swizzle_scale_factors(scales)
                    ),
                )
            else:
                shared_output[i] = layer.mlp._forward_shared_experts(local_input[i])
        local_input[i] = local_quant[i] = None

    def finish(i):
        a, b = bounds[i]
        main.wait_event(rs_done[i])
        if shared_output[i] is not None:
            output[a:b].add_(shared_output[i])
            shared_output[i] = None
        final = boundary.HcUpdate(layer, False).post(output[a:b], ffn_state[i])
        ffn_state[i] = None
        # Capture finalized residuals before a next-layer read consumes HcState.
        results.append(final.residual if isinstance(final, mhc.HcState) else final)
        if isinstance(final, mhc.HcState):
            pre_weights.append(final.pre)
        if lookahead:
            read = boundary.HcRead(next_layer, True).bind(batch, call)
            x, read_state = read.read(final, next_layer.input_layernorm)
            attention = next_layer.self_attn
            qkv, _ = attention.wqkv_a(x)
            q_lora, q_input = attention._normalize_q_lora(
                qkv[..., : attention.q_lora_rank]
            )
            # Match the normal query output dtype, including uniform FP8 pools.
            q_out = None
            if attention.uses_uniform_fp8_query():
                q_out = torch.empty(
                    (b - a, attention.n_local_heads, attention.head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=x.device,
                )
            q = attention._compute_q_b(q_input, positions[a:b], q_out)
            x_parts.append(x)
            query_parts.append((qkv, q_lora, q))
            next_states.append(read_state)

    submit_row_pipeline(
        count,
        prepare=prepare,
        gather=gather,
        compute=compute,
        scatter=scatter,
        shared=shared,
        finish=finish,
    )
    residual = torch.cat(results, dim=0)
    final = (
        mhc.HcState(residual, torch.cat(pre_weights, dim=0))
        if pre_weights
        else residual
    )
    if lookahead:

        def joined(name):
            values = [getattr(s, name) for s in next_states]
            return None if values[0] is None else torch.cat(values, dim=0)

        stats = None
        if next_states[0].stats is not None:
            stats = tuple(
                torch.cat([s.stats[j] for s in next_states]) for j in range(3)
            )
        read_state = boundary.ReadState(
            residual, joined("post"), joined("comb"), stats=stats, call=call
        )
        next_layer._query_lookahead = QueryLookahead(
            batch,
            final,
            positions,
            torch.cat(x_parts),
            *(torch.cat([p[j] for p in query_parts]) for j in range(3)),
            read_state,
        )
    return residual_batch.set_written(final, batch)
