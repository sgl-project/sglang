# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron VoiceChat text/ASR stage for online SGLang inference.

The acoustic perception encoder remains in NeMo. This module consumes its
per-frame embedding and extends SGLang's existing Nemotron-H implementation
with the parallel autoregressive ASR stream used by VoiceChat.

Adapted from NVIDIA NeMo's vLLM-Omni NemotronDuplexH implementation.
"""

from collections.abc import Iterable

import torch

from sglang.srt.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.srt.utils import add_prefix


class NemotronDuplexHForCausalLM(NemotronHForCausalLM):
    """Nemotron-H with acoustic input plus parallel text and ASR outputs."""

    def __init__(self, *, config, quant_config=None, prefix: str = ""):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.embed_asr_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_asr_tokens", prefix),
        )
        self.asr_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=add_prefix("asr_head", prefix),
        )
        self.pad_token_id = int(config.pad_token_id)
        self.register_buffer("_pad_combined_emb", None, persistent=False)

    def _combined_embeddings(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        custom_inputs = forward_batch.custom_inputs
        if custom_inputs is None or any(item is None for item in custom_inputs):
            raise ValueError(
                "NemotronDuplexH requires custom_inputs for every request."
            )

        combined = self.model.embed_tokens(input_ids)
        lengths = (
            forward_batch.extend_seq_lens_cpu
            if forward_batch.extend_seq_lens_cpu is not None
            else [1] * forward_batch.batch_size
        )
        offset = 0
        for item, length in zip(custom_inputs, lengths, strict=True):
            end = offset + length
            if item.get("is_system_prompt", False):
                if self._pad_combined_emb is None:
                    pad = torch.tensor([self.pad_token_id], device=input_ids.device)
                    self._pad_combined_emb = (
                        (self.model.embed_tokens(pad) + self.embed_asr_tokens(pad))
                        .squeeze(0)
                        .detach()
                    )
                combined[offset:end] += self._pad_combined_emb
            else:
                asr_ids = torch.as_tensor(
                    item["input_asr_ids"], device=input_ids.device, dtype=torch.long
                ).reshape(-1)
                if asr_ids.numel() == 1 and length != 1:
                    asr_ids = asr_ids.expand(length)
                acoustic = torch.as_tensor(
                    item["acoustic_embedding"],
                    device=input_ids.device,
                    dtype=combined.dtype,
                ).reshape(length, -1)
                if asr_ids.numel() != length:
                    raise ValueError("input_asr_ids must align with scheduled tokens.")
                if acoustic.shape != combined[offset:end].shape:
                    raise ValueError(
                        "acoustic_embedding must have shape "
                        f"({length}, {combined.shape[-1]})."
                    )
                combined[offset:end] += self.embed_asr_tokens(asr_ids) + acoustic
            offset = end
        return combined

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ):
        if input_embeds is not None:
            raise ValueError("NemotronDuplexH does not accept input_embeds.")
        combined = self._combined_embeddings(input_ids, forward_batch)
        hidden_states = self.model.forward(
            input_ids, positions, forward_batch, pp_proxy_tensors, combined
        )
        if not self.pp_group.is_last_rank:
            return hidden_states

        output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        asr_output = self.logits_processor(
            input_ids, hidden_states, self.asr_head, forward_batch
        )
        asr_tokens = torch.argmax(asr_output.next_token_logits, dim=-1)
        output.customized_info = {"asr_tokens": asr_tokens.tolist()}
        return output

    @staticmethod
    def _map_voicechat_weight_name(name: str) -> str:
        mappings = (
            ("stt_model.llm.backbone.", "model."),
            ("stt_model.llm.", "model."),
            ("stt_model.embed_tokens.", "model.embed_tokens."),
            ("stt_model.embed_asr_tokens.", "embed_asr_tokens."),
            ("stt_model.lm_head.", "lm_head."),
            ("stt_model.asr_head.", "asr_head."),
        )
        for old, new in mappings:
            if name.startswith(old):
                return new + name[len(old) :]
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        super().load_weights(
            (self._map_voicechat_weight_name(name), weight) for name, weight in weights
        )
        pad = torch.tensor(
            [self.pad_token_id], device=self.model.embed_tokens.weight.device
        )
        self._pad_combined_emb = (
            (self.model.embed_tokens(pad) + self.embed_asr_tokens(pad))
            .squeeze(0)
            .detach()
        )


EntryClass = NemotronDuplexHForCausalLM
