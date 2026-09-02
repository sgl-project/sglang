from collections.abc import Sequence
from typing import Any

import msgspec
import torch

from sglang.srt.sampling.watermark.config import WatermarkRegistry
from sglang.srt.sampling.watermark.request import WatermarkRequestConfig
from sglang.srt.sampling.watermark.textseal import (
    context_from_token_ids,
    request_nonce,
)


class WatermarkBatchInfo(msgspec.Struct, frozen=True, kw_only=True):
    enabled: torch.Tensor
    all_enabled: bool
    key_a: torch.Tensor
    key_b: torch.Tensor
    mixing_probabilities: torch.Tensor
    ngrams: torch.Tensor
    contexts: torch.Tensor
    nonces: torch.Tensor

    @classmethod
    def disabled(
        cls,
        batch_size: int,
        *,
        context_width: int,
        device: torch.device | str,
    ) -> "WatermarkBatchInfo":
        return cls(
            enabled=torch.zeros(batch_size, dtype=torch.bool, device=device),
            all_enabled=False,
            key_a=torch.zeros(batch_size, dtype=torch.long, device=device),
            key_b=torch.zeros(batch_size, dtype=torch.long, device=device),
            mixing_probabilities=torch.zeros(
                batch_size, dtype=torch.float, device=device
            ),
            ngrams=torch.ones(batch_size, dtype=torch.int32, device=device),
            contexts=torch.zeros(
                (batch_size, context_width), dtype=torch.long, device=device
            ),
            nonces=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def advance_contexts(
        self, token_ids: torch.Tensor, *, row_start: int = 0
    ) -> "WatermarkBatchInfo":
        contexts = self.contexts.clone()
        contexts[row_start:] = torch.cat(
            (contexts[row_start:, 1:], token_ids.long().view(-1, 1)), dim=1
        )
        return WatermarkBatchInfo(
            enabled=self.enabled,
            all_enabled=self.all_enabled,
            key_a=self.key_a,
            key_b=self.key_b,
            mixing_probabilities=self.mixing_probabilities,
            ngrams=self.ngrams,
            contexts=contexts,
            nonces=self.nonces,
        )

    def refresh_contexts(self, requests: Sequence[Any]) -> "WatermarkBatchInfo":
        contexts = torch.zeros_like(self.contexts)
        for row, request in enumerate(requests):
            if not self.enabled[row]:
                continue
            ngram = int(self.ngrams[row].item())
            token_ids = request.origin_input_ids + request.output_ids
            contexts[row, -ngram:] = context_from_token_ids(
                token_ids, ngram, device=contexts.device
            )
        return WatermarkBatchInfo(
            enabled=self.enabled,
            all_enabled=self.all_enabled,
            key_a=self.key_a,
            key_b=self.key_b,
            mixing_probabilities=self.mixing_probabilities,
            ngrams=self.ngrams,
            contexts=contexts,
            nonces=self.nonces,
        )

    def filter(self, keep_indices: torch.Tensor) -> "WatermarkBatchInfo":
        return WatermarkBatchInfo(
            enabled=self.enabled[keep_indices],
            all_enabled=self.all_enabled,
            key_a=self.key_a[keep_indices],
            key_b=self.key_b[keep_indices],
            mixing_probabilities=self.mixing_probabilities[keep_indices],
            ngrams=self.ngrams[keep_indices],
            contexts=self.contexts[keep_indices],
            nonces=self.nonces[keep_indices],
        )

    def merge(self, other: "WatermarkBatchInfo") -> "WatermarkBatchInfo":
        context_width = max(self.contexts.shape[1], other.contexts.shape[1])
        return WatermarkBatchInfo(
            enabled=torch.cat((self.enabled, other.enabled)),
            all_enabled=self.all_enabled and other.all_enabled,
            key_a=torch.cat((self.key_a, other.key_a)),
            key_b=torch.cat((self.key_b, other.key_b)),
            mixing_probabilities=torch.cat(
                (self.mixing_probabilities, other.mixing_probabilities)
            ),
            ngrams=torch.cat((self.ngrams, other.ngrams)),
            contexts=torch.cat(
                (
                    _left_pad_contexts(self.contexts, context_width),
                    _left_pad_contexts(other.contexts, context_width),
                )
            ),
            nonces=torch.cat((self.nonces, other.nonces)),
        )


def _left_pad_contexts(contexts: torch.Tensor, width: int) -> torch.Tensor:
    if contexts.shape[1] == width:
        return contexts
    padding = torch.zeros(
        (contexts.shape[0], width - contexts.shape[1]),
        dtype=contexts.dtype,
        device=contexts.device,
    )
    return torch.cat((padding, contexts), dim=1)


def build_watermark_batch_info(
    requests: Sequence[Any],
    registry: WatermarkRegistry,
    *,
    device: torch.device | str,
) -> WatermarkBatchInfo | None:
    if not any(
        isinstance(request.watermark, WatermarkRequestConfig) for request in requests
    ):
        return None

    configs = [
        (
            registry.resolve_request(request.watermark)
            if isinstance(request.watermark, WatermarkRequestConfig)
            else None
        )
        for request in requests
    ]
    max_ngram = max(config.ngram for config in configs if config is not None)
    contexts = torch.zeros((len(requests), max_ngram), dtype=torch.long, device=device)
    for row, (request, config) in enumerate(zip(requests, configs)):
        if config is None:
            continue
        token_ids = request.origin_input_ids + request.output_ids
        contexts[row, -config.ngram :] = context_from_token_ids(
            token_ids, config.ngram, device=device
        )

    return WatermarkBatchInfo(
        enabled=torch.tensor(
            [config is not None for config in configs],
            dtype=torch.bool,
            device=device,
        ),
        all_enabled=all(config is not None for config in configs),
        key_a=torch.tensor(
            [config.key_a if config is not None else 0 for config in configs],
            dtype=torch.long,
            device=device,
        ),
        key_b=torch.tensor(
            [config.key_b if config is not None else 0 for config in configs],
            dtype=torch.long,
            device=device,
        ),
        mixing_probabilities=torch.tensor(
            [
                config.mixing_probability if config is not None else 0.0
                for config in configs
            ],
            dtype=torch.float,
            device=device,
        ),
        ngrams=torch.tensor(
            [config.ngram if config is not None else 1 for config in configs],
            dtype=torch.int32,
            device=device,
        ),
        contexts=contexts,
        nonces=torch.tensor(
            [request_nonce(request.rid) for request in requests],
            dtype=torch.long,
            device=device,
        ),
    )
