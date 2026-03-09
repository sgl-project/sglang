"""SGLang Dream model (d3LLM-Dream, Qwen2.5-7B with full bidirectional attention)."""

from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen2 import Qwen2ForCausalLM


class DreamModel(Qwen2ForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        # dLLM needs full logits over all tokens (not just the last one).
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)
        # Dream uses full bidirectional (non-causal) attention; patch all layers.
        for layer in self.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "attn"):
                layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank and not get_embedding:
            # Dream's logits are next-token style: logits[i] predicts position i+1.
            # dLLM algorithms expect logits[i] to predict position i.
            # Right-shift hidden_states per sequence so the logits align correctly.
            if forward_batch.forward_mode.is_dllm_extend():
                seq_lens = forward_batch.extend_seq_lens_cpu
                if seq_lens is not None:
                    # Variable-length sequences (real forward pass)
                    parts = hidden_states.split(seq_lens)
                else:
                    # CUDA graph capture: all sequences are equal length
                    bs = forward_batch.batch_size
                    parts = hidden_states.view(bs, -1, hidden_states.shape[-1])
                hidden_states = torch.cat(
                    [torch.cat([p[:1], p[:-1]], dim=0) for p in parts], dim=0
                )

            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )

        if get_embedding:
            return self.pooler(hidden_states, forward_batch)
        return hidden_states


EntryClass = DreamModel
