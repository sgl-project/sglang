from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.runtime_context import get_parallel


class DreamModel(Qwen2ForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        if get_parallel().tp_size != 1:
            raise ValueError("DreamModel currently only supports TP=1")
        super().__init__(config, quant_config, prefix)

        if self.pp_group.world_size != 1:
            raise ValueError("DreamModel currently only supports PP=1")

        self.logits_processor = LogitsProcessor(config, return_full_logits=True)
        for layer in self.model.layers:
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
    ):
        assert not self.capture_aux_hidden_states

        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if not get_embedding:
            if forward_batch.forward_mode.is_dllm_extend():
                seq_lens = forward_batch.extend_seq_lens_cpu

                if seq_lens is None:
                    raise RuntimeError("Dream requires per-request sequence lengths")

                # Prefill BCG replays the transformer body at a padded token
                # bucket, while Dream's ragged canvas metadata still describes
                # only the real tokens.  The padding is appended after the
                # live canvas, so remove it before splitting by request.
                raw_num_tokens = sum(seq_lens)
                if raw_num_tokens > hidden_states.shape[0]:
                    raise RuntimeError(
                        "Dream sequence lengths exceed the hidden-state rows: "
                        f"{raw_num_tokens} > {hidden_states.shape[0]}"
                    )
                hidden_states = hidden_states[:raw_num_tokens]

                parts = hidden_states.split(seq_lens)

                hidden_states = torch.cat(
                    [
                        torch.cat(
                            [part[:1], part[:-1]],
                            dim=0,
                        )
                        for part in parts
                    ],
                    dim=0,
                )
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )

        return self.pooler(hidden_states, forward_batch)


EntryClass = DreamModel
