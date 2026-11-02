# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Adjacent causal injection adapted from Tencent/DSpine and vLLM PR #63704.

import weakref

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlashDecoderLayer,
    DFlashDraftModel,
    candidate_topk,
)
from sglang.srt.speculative.dspine_components import (
    DSpineCodeBridge,
    resolve_dspine_head_topk,
    score_dspine_transitions,
)


class DSpineDecoderLayer(DFlashDecoderLayer):
    def forward(
        self, positions, hidden_states, forward_batch, residual, input_normed=False
    ):
        if input_normed:
            if residual is None:
                raise RuntimeError("DSPINE pre-normalized input requires its residual")
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class DSpineSelector(CandidateSelector):
    def __init__(self, model):
        # Only reuse the common path walk, not DFlash2's learned selector head.
        nn.Module.__init__(self)
        self.top_k = resolve_dspine_head_topk(model.config)
        self._draft_ref = weakref.ref(model)

    def build_lattice(
        self, *, candidate_ids, unary_logits, hidden_states, anchor_token_ids
    ):
        model = self._draft_ref()
        if model is None or model.last_write_stash is None:
            raise RuntimeError(
                "DSPINE lattice requires the current draft forward state"
            )
        if not model._refinement_ready:
            raise RuntimeError(
                "DSPINE target head must be prepared before graph capture"
            )
        raw, rms, query = model.last_write_stash
        batch, steps, _ = candidate_ids.shape
        block = model.block_size
        if steps != block - 1 or raw.shape[0] != batch * block:
            raise ValueError(
                "DSPINE refinement state must match the current request blocks"
            )
        return score_dspine_transitions(
            candidate_ids,
            raw.view(batch, block, -1)[:, 1:],
            rms.view(batch, block)[:, 1:],
            query.view(batch, block, -1)[:, 1:],
            anchor_token_ids,
            codebook=model.selected_prefix_bridge.codebook,
            pred_gate=model.pred_gate_f32,
            prev=model.prev_f32,
            gate_bias=model.gate_bias_f32,
            write=model.write_f32,
            norm=model.norm_f32,
            lm_head_weight=model.lm_head.weight,
            norm_eps=model.norm.variance_epsilon,
            write_gram=model.write_gram_f32,
        )


class DSpineDraftModel(DFlashDraftModel):
    decoder_layer_cls = DSpineDecoderLayer

    def __init__(self, config, quant_config=None, prefix=""):
        if quant_config is not None:
            raise ValueError("DSPINE draft quantization is not supported")
        if getattr(config, "is_causal", None) is not True:
            raise ValueError("DSPINE requires explicit causal attention")
        if getattr(config, "use_gemma_norm", False):
            raise ValueError("DSPINE requires Qwen3 RMSNorm")
        super().__init__(config, quant_config=None, prefix=prefix)
        if (
            self.draft_config.is_mla
            or self.draft_config.conv_kernel_size
            or self.projector_type not in (None, "linear")
        ):
            raise ValueError(
                "DSPINE requires the dense Qwen3 backbone and linear context projector"
            )
        if (
            self.draft_config.output_multiplier != 1.0
            or self.draft_config.final_logit_softcapping is not None
        ):
            raise ValueError("DSPINE requires an unscaled linear output head")
        self.selected_prefix_bridge = DSpineCodeBridge(
            config, params_dtype=self.fc.weight.dtype
        )
        self.candidate_selector = DSpineSelector(self)
        self.lm_head = None
        self.last_write_stash = None
        self._refinement_ready = False
        # Never silently serve a partially loaded backbone.
        for name, param in self.named_parameters():
            if not name.startswith("selected_prefix_bridge."):
                param.data.fill_(float("nan"))

    @torch.no_grad()
    def prepare_refinement(self, lm_head):
        weight = getattr(lm_head, "weight", None)
        shape = (int(self.config.vocab_size), int(self.config.hidden_size))
        if weight is None or tuple(weight.shape) != shape or weight.stride(1) != 1:
            raise ValueError(
                "DSPINE requires a complete row-contiguous target output head"
            )
        if weight.dtype != self.selected_prefix_bridge.codebook.dtype:
            raise ValueError(
                "DSPINE target head and draft activations must use the same dtype"
            )
        if (
            weight.dtype not in (torch.bfloat16, torch.float32)
            or getattr(lm_head, "quant_method", None) is not None
        ):
            # Ordinary unquantized heads may carry an UnquantizedEmbeddingMethod.
            from sglang.srt.layers.logits_processor import (
                should_apply_lm_head_quant_method,
            )

            if weight.dtype not in (
                torch.bfloat16,
                torch.float32,
            ) or should_apply_lm_head_quant_method(
                lm_head, getattr(lm_head, "quant_method", None)
            ):
                raise ValueError("DSPINE requires an unquantized BF16/FP32 target head")
        if self._refinement_ready and self.lm_head is lm_head:
            return
        self.lm_head = lm_head
        bridge = self.selected_prefix_bridge
        for name, tensor in (
            ("pred_gate_f32", bridge.predecessor_gate.weight),
            ("prev_f32", bridge.prev.weight),
            ("gate_bias_f32", bridge.gate.bias),
            ("write_f32", bridge.feature_write_weights[-1]),
            ("norm_f32", self.norm.weight),
            (
                "write_gram_f32",
                bridge.feature_write_weights[-1].float().T
                @ bridge.feature_write_weights[-1].float(),
            ),
        ):
            value = tensor.detach().float().contiguous()
            existing = getattr(self, name, None)
            if existing is None:
                self.register_buffer(name, value, persistent=False)
            else:
                if existing.shape != value.shape:
                    raise ValueError(
                        "DSPINE weight reload cannot change refinement geometry"
                    )
                existing.copy_(value)
        self._refinement_ready = True

    @torch.no_grad()
    def compute_candidates(self, hidden):
        if self.lm_head is None:
            raise RuntimeError("DSPINE target output head is unset")
        ids, logits, _ = candidate_topk(
            hidden, self.lm_head, self.candidate_selector.top_k
        )
        ids, order = ids.sort(dim=-1)
        return ids, logits.gather(-1, order)

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        if input_embeds is None:
            if not hasattr(self, "forward_embed"):
                raise ValueError("DSPINE requires target input embeddings")
            input_embeds = self.forward_embed(input_ids)
        if input_ids.numel() % self.block_size:
            raise ValueError("DSPINE requires complete trained request blocks")
        if input_ids.numel() == 0:
            self.last_write_stash = None
            return LogitsProcessorOutput(
                next_token_logits=None, hidden_states=input_embeds
            )
        hidden, residual = input_embeds, None
        input_normed = False
        self.last_write_stash = None
        bridge = self.selected_prefix_bridge
        for index, layer in enumerate(self.layers):
            hidden, residual = layer(
                positions, hidden, forward_batch, residual, input_normed=input_normed
            )
            input_normed = False
            if index in bridge.active_layers:
                next_norm = (
                    self.layers[index + 1].input_layernorm
                    if index + 1 < len(self.layers)
                    else self.norm
                )
                hidden, residual, raw, rms, packed = bridge(
                    hidden, residual, input_ids, index, next_norm
                )
                input_normed = True
                if index == len(self.layers) - 1:
                    self.last_write_stash = (raw, rms, packed[:, : bridge.rank])
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    def load_weights(self, weights):
        self._refinement_ready = False
        bridge = self.selected_prefix_bridge
        bridge.begin_load()

        def backbone_weights():
            for name, tensor in weights:
                short = name.removeprefix("model.")
                if short.startswith("selected_prefix_bridge."):
                    bridge.load_weight(
                        short.removeprefix("selected_prefix_bridge."), tensor
                    )
                elif short in (
                    "draft_block_causal_version",
                    "last_write_refine_version",
                ):
                    if int(tensor.item()) != 1:
                        raise ValueError(
                            f"Unsupported DSPINE checkpoint version: {short}"
                        )
                else:
                    yield name, tensor

        super().load_weights(backbone_weights())

    def post_load_weights(self):
        super().post_load_weights()
        self.selected_prefix_bridge.prepare()
        missing = [
            name
            for name, param in self.named_parameters()
            if not name.startswith(("selected_prefix_bridge.", "lm_head."))
            and not torch.isfinite(param).all().item()
        ]
        if missing:
            raise ValueError(
                f"Missing or non-finite DSPINE backbone weights: {missing}"
            )
        if self.lm_head is not None:
            self._refinement_ready = False
            self.prepare_refinement(self.lm_head)


EntryClass = DSpineDraftModel
