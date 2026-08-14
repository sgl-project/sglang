# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nemotron VoiceChat thinker stage for online SGLang inference.

The acoustic perception encoder remains in NeMo. This module consumes its
per-frame embedding and extends SGLang's existing Nemotron-H implementation
with the parallel autoregressive function-token stream used by VoiceChat.

Adapted from NVIDIA NeMo's vLLM-Omni NemotronDuplexH implementation.
"""

from collections.abc import Iterable

import torch

from sglang.srt.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.srt.utils import add_prefix


class NemotronDuplexHForCausalLM(NemotronHForCausalLM):
    """Nemotron-H with acoustic input plus text and function-token outputs."""

    def __init__(self, *, config, quant_config=None, prefix: str = ""):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        if bool(getattr(config, "predict_user_text", False)):
            raise NotImplementedError(
                "NemotronDuplexH does not implement predict_user_text=True; "
                "the published VoiceChat checkpoint disables that channel."
            )
        if not bool(getattr(config, "use_function_head", False)):
            raise ValueError(
                "NemotronDuplexH requires use_function_head=True for the "
                "published VoiceChat checkpoint."
            )
        if getattr(config, "fuse_method", "add") != "add":
            raise NotImplementedError(
                "NemotronDuplexH only supports fuse_method='add'."
            )
        self.function_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=add_prefix("function_head", prefix),
        )
        self.text_channel_weight = float(config.duplex_text_channel_weight)
        self.user_channel_weight = float(config.duplex_user_channel_weight)
        self.function_channel_weight = float(config.duplex_function_channel_weight)
        self.pad_token_id = int(config.pad_token_id)

    def _combined_embeddings(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        custom_inputs = forward_batch.custom_inputs
        if custom_inputs is None or any(item is None for item in custom_inputs):
            raise ValueError(
                "NemotronDuplexH requires custom_inputs for every request."
            )

        combined = torch.empty(
            (input_ids.shape[0], self.config.hidden_size),
            device=input_ids.device,
            dtype=self.model.embed_tokens.weight.dtype,
        )
        lengths = (
            forward_batch.extend_seq_lens_cpu
            if forward_batch.extend_seq_lens_cpu is not None
            else [1] * forward_batch.batch_size
        )
        offset = 0
        for item, length in zip(custom_inputs, lengths, strict=True):
            end = offset + length
            if item.get("is_initial_prefill", False):
                prompt_length = int(item["prompt_length"])
                if length != prompt_length + 1:
                    raise ValueError(
                        "Initial VoiceChat prefill must contain the prompt plus "
                        "one acoustic-frame placeholder."
                    )
                acoustic = torch.as_tensor(
                    item["acoustic_embedding"],
                    device=input_ids.device,
                    dtype=combined.dtype,
                ).reshape(1, -1)
                if acoustic.shape[-1] != combined.shape[-1]:
                    raise ValueError(
                        "acoustic_embedding hidden size does not match the model."
                    )
                timeline = torch.cat(
                    [self.model.embed_tokens(input_ids[offset : end - 1]), acoustic],
                    dim=0,
                )
                pad_ids = torch.full(
                    (length,), self.pad_token_id, device=input_ids.device
                )
                pad_embeddings = self.model.embed_tokens(pad_ids)
                combined[offset:end] = (
                    pad_embeddings * self.text_channel_weight
                    + timeline * self.user_channel_weight
                    + pad_embeddings * self.function_channel_weight
                )
            else:
                function_ids = torch.as_tensor(
                    item["input_function_ids"],
                    device=input_ids.device,
                    dtype=torch.long,
                ).reshape(-1)
                if function_ids.numel() == 1 and length != 1:
                    function_ids = function_ids.expand(length)
                acoustic = torch.as_tensor(
                    item["acoustic_embedding"],
                    device=input_ids.device,
                    dtype=combined.dtype,
                ).reshape(length, -1)
                if function_ids.numel() != length:
                    raise ValueError(
                        "input_function_ids must align with scheduled tokens."
                    )
                if acoustic.shape != combined[offset:end].shape:
                    raise ValueError(
                        "acoustic_embedding must have shape "
                        f"({length}, {combined.shape[-1]})."
                    )
                combined[offset:end] = (
                    self.model.embed_tokens(input_ids[offset:end])
                    * self.text_channel_weight
                    + acoustic * self.user_channel_weight
                    + self.model.embed_tokens(function_ids)
                    * self.function_channel_weight
                )
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
        function_output = self.logits_processor(
            input_ids, hidden_states, self.function_head, forward_batch
        )
        function_tokens = torch.argmax(function_output.next_token_logits, dim=-1)
        output.customized_info = {"function_tokens": function_tokens.tolist()}
        return output

    @staticmethod
    def _map_voicechat_weight_name(name: str) -> str:
        mappings = (
            ("stt_model.llm.backbone.", "model."),
            ("stt_model.llm.", "model."),
            ("stt_model.embed_tokens.", "model.embed_tokens."),
            ("stt_model.lm_head.", "lm_head."),
            ("stt_model.function_head.", "function_head."),
        )
        for old, new in mappings:
            if name.startswith(old):
                return new + name[len(old) :]
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        saw_function_head = False

        def mapped_weights():
            nonlocal saw_function_head
            for name, weight in weights:
                if name == "stt_model.function_head.weight":
                    saw_function_head = True
                yield self._map_voicechat_weight_name(name), weight

        super().load_weights(mapped_weights())
        if not saw_function_head:
            raise ValueError(
                "Checkpoint is missing 'stt_model.function_head.weight'; "
                "reconvert it with the VoiceChat Duplex converter."
            )


EntryClass = NemotronDuplexHForCausalLM
