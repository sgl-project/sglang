# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Nemotron Decision: a bidirectional Ministral3 with a vocabulary readout."""

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.models.ministral3 import Ministral3ForCausalLM, Ministral3Model
from sglang.srt.models.utils import WeightsMapper


class PackedDecisionAttention:
    def __init__(self):
        self.active = False
        self.wrapper = None
        self.needs_plan = False

    def prepare(self, forward_batch, positions):
        layouts = forward_batch.decision_layouts
        self.active = layouts is not None and any(
            layout is not None for layout in layouts
        )
        if not self.active:
            return
        query_ptr, key_ptr, gather, reset_positions = [0], [0], [], []
        offset = 0
        for length, layout in zip(forward_batch.extend_seq_lens_cpu, layouts):
            if layout is None:
                segments = [(length, [(offset, offset + length)])]
                reset_positions.extend(range(length))
            else:
                if sum(layout) != length:
                    raise RuntimeError("packed Decision requires a complete prefill")
                shared, *questions = layout
                segments = [(shared, [(offset, offset + shared)])]
                reset_positions.extend(range(shared))
                at = offset + shared
                for question_length in questions:
                    segments.append(
                        (
                            question_length,
                            [(offset, offset + shared), (at, at + question_length)],
                        )
                    )
                    reset_positions.extend(range(shared, shared + question_length))
                    at += question_length
            for query_length, ranges in segments:
                query_ptr.append(query_ptr[-1] + query_length)
                for start, end in ranges:
                    gather.extend(range(start, end))
                key_ptr.append(len(gather))
            offset += length
        self.num_tokens = offset
        self.query_ptr_cpu = torch.tensor(query_ptr, dtype=torch.int32)
        self.key_ptr_cpu = torch.tensor(key_ptr, dtype=torch.int32)
        self.kv_indices = torch.tensor(
            gather, dtype=torch.int32, device=positions.device
        )
        self.last_page_len = torch.ones(len(query_ptr) - 1, dtype=torch.int32)
        positions[:offset].copy_(
            torch.tensor(
                reset_positions, dtype=positions.dtype, device=positions.device
            )
        )
        self.needs_plan = True

    def run(self, q, k, v, layer):
        if self.wrapper is None:
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper

            self.workspace = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=q.device,
            )
            self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace, "NHD", backend="auto"
            )
        if self.needs_plan:
            self.wrapper.plan(
                self.query_ptr_cpu,
                self.key_ptr_cpu,
                self.kv_indices,
                self.last_page_len,
                layer.tp_q_head_num,
                layer.tp_k_head_num,
                layer.head_dim,
                page_size=1,
                causal=False,
                sm_scale=layer.scaling,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )
            self.needs_plan = False
        q = q[: self.num_tokens].view(-1, layer.tp_q_head_num, layer.head_dim)
        # A one-token page view lets questions share state K/V without copying it.
        cache = (
            k.view(-1, 1, layer.tp_k_head_num, layer.head_dim),
            v.view(-1, 1, layer.tp_v_head_num, layer.v_head_dim),
        )
        return self.wrapper.run(q, cache).reshape(self.num_tokens, -1)


class NemotronDecisionRadixAttention(RadixAttention):
    def __init__(self, *, packed, **kwargs):
        super().__init__(**kwargs)
        self.packed = packed

    def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        if self.packed.active:
            return self.packed.run(q, k, v, self)
        return super().forward(q, k, v, forward_batch, save_kv_cache, **kwargs)


class NemotronForDecision(Ministral3ForCausalLM):
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={"encoder.": "model.", "diffusion_head.": "lm_head."}
    )

    def _init_model(self, config, quant_config=None, prefix=""):
        if config.is_causal or config.sliding_window is not None:
            raise ValueError("Nemotron Decision requires full bidirectional attention")
        return NemotronDecisionModel(
            config=config, quant_config=quant_config, prefix=prefix
        )

    def load_weights(self, weights):
        return super().load_weights(self.hf_to_sglang_mapper.apply(weights))


class NemotronDecisionModel(Ministral3Model):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.packed_attention = PackedDecisionAttention()
        for layer in self.layers[self.start_layer : self.end_layer]:
            attention = layer.self_attn
            old = attention.attn
            attention.attn = NemotronDecisionRadixAttention(
                packed=self.packed_attention,
                num_heads=old.tp_q_head_num,
                num_kv_heads=old.tp_k_head_num,
                head_dim=old.head_dim,
                scaling=old.scaling,
                layer_id=old.layer_id,
                attn_type=AttentionType.ENCODER_ONLY,
                quant_config=quant_config,
                prefix=add_prefix(f"layers.{old.layer_id}.self_attn.attn", prefix),
            )
            # Query scaling uses the training window, not the extended context limit.
            attention.max_position_embeddings = config.rope_parameters.get(
                "original_max_position_embeddings", config.max_position_embeddings
            )

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        self.packed_attention.active = False
        if forward_batch.decision_layouts is not None:
            positions = positions.clone()
            self.packed_attention.prepare(forward_batch, positions)
        return super().forward(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )


EntryClass = NemotronForDecision
