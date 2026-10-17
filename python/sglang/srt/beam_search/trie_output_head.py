# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Compact SID projection with streamed full-vocabulary normalization."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import msgspec
import torch

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.environ import envs

_DEFAULT_NORMALIZER_CHUNK_SIZE = 8192
logger = logging.getLogger(__name__)


class BeamTrieHeadOutput(msgspec.Struct, frozen=True):
    logits: torch.Tensor
    normalizer: Optional[torch.Tensor]


def use_topk_logprob() -> bool:
    return envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.get()


def configure_model_trie_output_heads(
    *,
    model: torch.nn.Module,
    config: Optional[TrieOutputHeadConfig],
    device: str,
    tp_size: int,
    quantization: Optional[str],
    lora_enabled: bool,
    rl_on_policy_target: Optional[str],
) -> bool:
    from sglang.srt.layers.logits_processor import LogitsProcessor

    enabled = config is not None and (
        device == "cuda"
        and tp_size == 1
        and quantization is None
        and not lora_enabled
        and rl_on_policy_target is None
    )
    if config is not None and not enabled:
        logger.info(
            "Using dense LM-head scoring for model-bundled beam trie; "
            "the compact output head requires CUDA, TP=1, an unquantized "
            "LM head, and no LoRA or RL target."
        )
    configured = 0
    for module in model.modules():
        if isinstance(module, LogitsProcessor):
            module.configure_beam_trie(config if enabled else None)
            configured += 1
    if enabled and configured == 0:
        raise RuntimeError(
            "A model-bundled beam trie was discovered, but the loaded "
            "generation model has no compatible LogitsProcessor"
        )
    return enabled


class BeamTrieOutputHead:
    """Projects the active SID codebook and optional full-vocabulary scores."""

    def __init__(
        self,
        config: TrieOutputHeadConfig,
        vocab_size: int,
        chunk_size: int = _DEFAULT_NORMALIZER_CHUNK_SIZE,
    ):
        self.config = config
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size

    def project_mixed(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        depths: torch.Tensor,
        project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        transform: Callable[[torch.Tensor], torch.Tensor],
        use_grouped_projection: bool = False,
    ) -> BeamTrieHeadOutput:
        start = self.config.token_start
        end = self.config.token_end
        logits = None
        if use_grouped_projection:
            from sglang.srt.beam_search.trie_kernels import (
                project_active_codebook_logits,
            )

            logits = project_active_codebook_logits(
                hidden_states,
                weight[start:end],
                depths,
                self.config.codebook_size,
            )
        if logits is None:
            all_logits = project(hidden_states, weight[start:end]).view(
                hidden_states.shape[0],
                self.config.num_codebooks,
                self.config.codebook_size,
            )
            row_indices = torch.arange(
                hidden_states.shape[0], device=hidden_states.device
            )
            logits = all_logits[row_indices, depths]
        normalizer = (
            None
            if use_topk_logprob()
            else self._stream_normalizer(
                hidden_states,
                weight,
                project,
                transform,
            )
        )
        return BeamTrieHeadOutput(logits, normalizer)

    def _stream_normalizer(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if self.vocab_size > weight.shape[0]:
            raise RuntimeError(
                f"Model vocabulary ({self.vocab_size}) exceeds LM-head rows "
                f"({weight.shape[0]})"
            )

        row_max = None
        row_log_sum = None
        for start in range(0, self.vocab_size, self.chunk_size):
            logical_end = min(start + self.chunk_size, self.vocab_size)
            chunk_logits = project(hidden_states, weight[start:logical_end])
            chunk_logits = transform(chunk_logits)
            if chunk_logits.is_cuda:
                from sglang.srt.layers.logsumexp import row_logsumexp

                chunk_max, chunk_log_sum = row_logsumexp(chunk_logits)
            else:
                chunk_logits = chunk_logits.float()
                chunk_max = chunk_logits.max(dim=-1).values
                chunk_log_sum = torch.log(
                    torch.exp(chunk_logits - chunk_max[:, None]).sum(dim=-1)
                )

            if row_max is None:
                row_max = chunk_max
                row_log_sum = chunk_log_sum
            else:
                next_max = torch.maximum(row_max, chunk_max)
                row_log_sum = torch.logaddexp(
                    row_log_sum + row_max - next_max,
                    chunk_log_sum + chunk_max - next_max,
                )
                row_max = next_max

        assert row_max is not None and row_log_sum is not None
        return torch.stack((row_max, row_log_sum), dim=1)
