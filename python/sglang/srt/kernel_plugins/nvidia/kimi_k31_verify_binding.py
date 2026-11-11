# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 verify megakernel host binding; only tensors cross into the plugin."""

from __future__ import annotations

import torch

from sglang.kernels.plugin_kernels.nvidia.spec import Step
from sglang.srt.layers.attention.linear.utils import (
    select_verify_intermediate_state_indices,
)
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel

# Packed MXFP4 routed-expert shards (w13, w2) per TP size.
_EXPERT_SHARDS = {
    8: ((896, 768, 1792), (896, 3584, 192)),
    16: ((896, 512, 1792), (896, 3584, 128)),
}


def covers_short_verify(forward_batch, query_width):
    # seq_lens_cpu already includes the verify block; adding another is conservative.
    lengths = forward_batch.seq_lens_cpu
    if lengths is None:
        return False
    return int(lengths.max()) + query_width <= 2048


class K31VerifyHostBinding:
    capture_stock_fallback = True

    def __init__(self, model):
        self.model = model
        self.aux = None
        self.capture_layers = ()

    def weights_view(self, context_length, *, query_width=8):
        # optional plugin: imported only when the megakernel is installed
        from sgl_nvidia_mega.kimi_k31.verify_binding import (
            VerifyLayerWeights,
            VerifyWeights,
        )

        lm = self.model.language_model
        backbone = lm.model
        layers = tuple(backbone.layers)
        if len(layers) != 93:
            raise ValueError("K31 verify requires the complete 93-layer model")
        tp = get_parallel().tp_size
        if tp not in _EXPERT_SHARDS:
            raise ValueError(f"K31 verify serves TP8/TP16, got TP{tp}")
        w13_shape, w2_shape = _EXPERT_SHARDS[tp]
        records = []
        full_ids = []
        linear_ids = []
        for layer in layers:
            m = layer.self_attn
            idx = int(layer.layer_idx)
            is_mla = hasattr(m, "q_b_proj")
            if is_mla:
                if (
                    m.indexer is None
                    or not m.indexer.skip_rope
                    or m.indexer.apply_hadamard
                ):
                    raise ValueError(
                        "K31 verify requires NoPE index-K on every MLA layer"
                    )
                parts = (m.fused_qkv_a_proj_with_mqa.weight, m.g_proj.weight)
                padding = 0
            else:
                parts = (m.fused_qkvg_proj.weight, m.f_a_proj.weight, m.b_proj.weight)
                # Pad rows to a multiple of 8, matching the stock layout.
                padding = -sum(p.shape[0] for p in parts) % 8
            # Stock Parameters become views of packed, which the module keeps alive.
            packed = torch.cat(
                tuple(p.detach() for p in parts)
                + (parts[0].new_zeros((padding, 7168)),),
                dim=0,
            )
            offset = 0
            for param in parts:
                rows = param.shape[0]
                param.data = packed[offset : offset + rows]
                offset += rows
            m.__dict__["_k31_verify_projection_storage"] = packed
            if is_mla:
                full_ids.append(idx)
                indexer = m.indexer
                a = (
                    packed,
                    m.q_a_layernorm.weight,
                    m.kv_a_layernorm.weight,
                    m.q_b_proj.weight,
                    m.w_kc,
                    m.w_vc,
                    m.o_proj.weight,
                    indexer.wk.weight,
                    indexer.k_norm.weight,
                    indexer.k_norm.bias,
                )

                def scalar(value):
                    return (
                        float(value.item())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )

                qm = m.attn_mqa
                sc = (
                    float(m.q_a_layernorm.variance_epsilon),
                    float(m.kv_a_layernorm.variance_epsilon),
                    scalar(m.scaling),
                    scalar(m.w_scale),
                    scalar(
                        1.0
                        if getattr(qm, "q_scale_float", None) is None
                        else qm.q_scale_float
                    ),
                    scalar(
                        1.0
                        if getattr(qm, "k_scale_float", None) is None
                        else qm.k_scale_float
                    ),
                    float(indexer.k_norm.variance_epsilon),
                )
            else:
                linear_ids.append(idx)
                # Keep the padding: the stock F_B GEMM needs 16-byte-aligned F_A rows.
                m.__dict__["_bfa_w"] = packed[parts[0].shape[0] :]
                a = (
                    packed,
                    m.qkv_conv1d.weight,
                    m.f_b_proj.weight,
                    m.A_log.reshape(-1),
                    m.dt_bias,
                    m.o_norm.weight,
                    m.o_proj.weight,
                )
                sc = (float(m.attn.lower_bound), float(m.o_norm.eps))
            res = (
                layer.self_attention_res_norm.weight,
                layer.self_attention_res_proj.weight,
                layer.input_layernorm.weight,
                layer.mlp_res_norm.weight,
                layer.mlp_res_proj.weight,
                layer.post_attention_layernorm.weight,
            )
            eps = (
                float(layer.self_attention_res_norm.variance_epsilon),
                float(layer.input_layernorm.variance_epsilon),
                float(layer.mlp_res_norm.variance_epsilon),
                float(layer.post_attention_layernorm.variance_epsilon),
            )
            mlp = layer.mlp
            if layer._is_moe_layer:
                e = mlp.experts
                if (
                    e.w13_weight.dtype != torch.uint8
                    or tuple(e.w13_weight.shape) != w13_shape
                ):
                    raise ValueError(
                        f"K31 verify requires TP{tp} packed MXFP4 routed experts"
                    )
                if tuple(e.w2_weight.shape) != w2_shape:
                    raise ValueError("K31 verify routed output shard geometry changed")
                ff = (
                    e.w13_weight,
                    e.w13_weight_scale,
                    e.w2_weight,
                    e.w2_weight_scale,
                    mlp.shared_experts.gate_up_proj.weight,
                    mlp.shared_experts.down_proj.weight,
                    mlp.gate.weight,
                    mlp.gate.e_score_correction_bias,
                    mlp.routed_expert_down_proj.weight,
                    mlp.routed_expert_up_proj.weight,
                    mlp.routed_expert_norm.weight,
                )
                fs = (float(mlp.routed_scaling_factor),)
            else:
                ff = (mlp.gate_up_proj.weight, mlp.down_proj.weight)
                fs = ()
            records.append(
                VerifyLayerWeights(
                    idx,
                    int(not is_mla),
                    int(layer._is_moe_layer),
                    a,
                    sc,
                    res,
                    eps,
                    ff,
                    fs,
                    int(layer.prev_valid_blocks),
                    int(layer.is_block_write_layer),
                )
            )
        if (len(full_ids), len(linear_ids)) != (24, 69):
            raise ValueError("K31 verify requires 24 MLA and 69 KDA layers")
        cfg = lm.config
        return VerifyWeights(
            tuple(records),
            backbone.embed_tokens.weight,
            (
                backbone.output_attn_res_norm.weight,
                backbone.output_attn_res_proj.weight,
                backbone.norm.weight,
            ),
            (
                float(backbone.output_attn_res_norm.variance_epsilon),
                float(backbone.norm.variance_epsilon),
            ),
            tuple(full_ids),
            tuple(linear_ids),
            int(cfg.attn_res_block_size),
            float(cfg.rms_norm_eps),
            float(cfg.activation_situ_beta),
            float(cfg.activation_situ_linear_beta),
            int(cfg.moe_renormalize),
            int(context_length),
            query_width=int(query_width),
            request_batches=(1, 2, 4) if int(query_width) == 8 else (1,),
        )

    def can_run_graph(self, built, forward_batch):
        return forward_batch.forward_mode.is_target_verify() and covers_short_verify(
            forward_batch, built.query_width
        )

    def configure_aux(self, built, layer_ids):
        ids = tuple(int(i) for i in layer_ids)
        if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= 93 for i in ids):
            raise ValueError("K31 verify aux requires distinct target layer outputs")
        width = len(ids) * 7168
        if self.model.get_aux_hidden_states_width() != width:
            raise ValueError(
                "K31 verify aux width disagrees with the target capture layout"
            )
        if self.aux is not None:
            if ids != self.capture_layers:
                raise RuntimeError("K31 verify aux layers changed after preparation")
            return
        device = next(self.model.parameters()).device
        self.capture_layers = ids
        self.aux = torch.empty(
            (max(built.widths), width), device=device, dtype=torch.bfloat16
        )
        built.configure_aux(ids)

    def pool_view(self, pool, req_pool):
        # optional plugin: imported only when the megakernel is installed
        from sgl_nvidia_mega.kimi_k31.verify_binding import (
            KdaVerifyPools,
            VerifyLayerPools,
            VerifyPoolView,
        )

        if req_pool.mamba_pool.replayssm_request_ring:
            raise ValueError(
                "K31 verify indexes the KDA ReplaySSM ring by mamba slot; "
                "SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT is not supported"
            )
        full = pool.full_kv_pool
        layers = []
        full_map = pool.full_attention_layer_id_mapping
        for layer_id in range(93):
            if layer_id in full_map:
                index = full_map[layer_id]
                cache = full.kv_buffer[index].view(
                    -1, pool.page_size, full.kv_cache_dim
                )
                index_cache = full.get_index_k_with_scale_buffer(index)
                layers.append(VerifyLayerPools(layer_id, (), (cache, index_cache)))
            else:
                state = req_pool.mamba_pool.mamba2_layer_cache(
                    req_pool.mamba_map[layer_id]
                )
                if state.intermediate_ssm is not None or state.replayssm_rawv is None:
                    raise ValueError(
                        "K31 verify requires KDA ReplaySSM speculative pools"
                    )
                kda = KdaVerifyPools(
                    state.temporal,
                    state.conv[0].transpose(-1, -2),
                    state.replayssm_rawv,
                    state.replayssm_rawk,
                    state.replayssm_g,
                    state.replayssm_beta,
                    state.intermediate_conv_window[0].transpose(-1, -2),
                )
                layers.append(VerifyLayerPools(layer_id, (kda,), ()))
        return VerifyPoolView(tuple(layers), int(pool.page_size), int(full.size))

    def plan(self, built, input_ids, forward_batch, *args, **kwargs):
        if not forward_batch.forward_mode.is_target_verify():
            return None
        if kwargs.get("get_embedding", args[0] if args else False):
            return None
        b = int(forward_batch.batch_size)
        r = int(input_ids.shape[0])
        if not b or r % b:
            return None
        q = r // b
        if not built.supports(b, q, r):
            return None
        if (
            forward_batch.spec_info is None
            or forward_batch.ragged_verify_layout is not None
        ):
            return None
        if not torch.cuda.is_current_stream_capturing() and not covers_short_verify(
            forward_batch, q
        ):
            return None
        return Step(mode="target_verify", rows=r)

    def forward(
        self, built, step, input_ids, positions, forward_batch, *args, **kwargs
    ):
        # optional plugin: imported only when the megakernel is installed
        from sgl_nvidia_mega.kimi_k31.verify_binding import (
            KdaVerifyStep,
            VerifyStepView,
        )

        backend = get_attn_backend()
        full = backend.full_attn_backend.forward_metadata
        linear = backend.linear_attn_backend
        metadata = linear.forward_metadata
        b = int(forward_batch.batch_size)
        q = step.rows // b
        slots = metadata.mamba_cache_indices[:b]
        scratch = select_verify_intermediate_state_indices(
            linear.verify_intermediate_state_indices,
            forward_batch.req_pool_indices,
            slots >= 0,
            linear.req_to_token_pool.size,
        )[:b]
        if self.aux is None:
            raise RuntimeError(
                "K31 verify aux capture was not configured before graph warmup"
            )
        kda = KdaVerifyStep(
            slots, scratch, (slots >= 0).to(torch.int32), b, q, step.rows
        )
        view = VerifyStepView(
            full.real_page_table[:b],
            full.cache_seqlens_int32[:b],
            forward_batch.out_cache_loc[: step.rows],
            kda,
            self.aux[: step.rows],
            b,
            q,
            step.rows,
        )
        lm = self.model.language_model
        embeddings = lm.model.embed_tokens(input_ids)
        hidden, aux = built.run_verify(embeddings, view)
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)
