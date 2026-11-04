"""Replay mixed target and draft transforms with staged, live request data."""

from __future__ import annotations

from bisect import bisect_left
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.watermarking.synthid.processor import SynthIDWatermarkProcessor


class _MixedTargetGraph:
    def __init__(
        self,
        *,
        processor: SynthIDWatermarkProcessor,
        batch_size: int,
        rows_per_request: int,
        vocab_size: int,
        stream,
        pool,
        sample_draft: bool,
    ):
        self.batch_size = batch_size
        self.rows_per_request = rows_per_request
        self.sample_draft = sample_draft
        device = processor.keys.device
        rows = batch_size * rows_per_request
        self.logits = torch.zeros(rows, vocab_size, dtype=torch.float32, device=device)
        self.temperatures = torch.ones(
            batch_size, 1, dtype=torch.float32, device=device
        )
        self.hashes = torch.zeros(rows, dtype=torch.int64, device=device)
        self.apply_mask = torch.ones(rows, dtype=torch.bool, device=device)
        self.enabled = torch.arange(batch_size, device=device) % 2 == 0
        self.graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self._transform(processor=processor)
            self._transform(processor=processor)
        stream.synchronize()
        with torch.cuda.graph(self.graph, stream=stream, pool=pool):
            self.output = self._transform(processor=processor)
        current.wait_stream(stream)

    def _transform(
        self, *, processor: SynthIDWatermarkProcessor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits.clone()
        processor._apply_with_hashes(
            logits,
            self.temperatures,
            self.hashes,
            self.apply_mask,
            batch_size=self.batch_size,
            rows_per_request=self.rows_per_request,
            scatter_to_logits=True,
            keys=processor.keys,
        )
        enabled = self.enabled.repeat_interleave(self.rows_per_request)[:, None]
        scaled = self.logits / self.temperatures.repeat_interleave(
            self.rows_per_request, dim=0
        )
        output = torch.where(enabled, logits, scaled)
        if not self.sample_draft:
            return output
        from sglang.srt.speculative.spec_utils import fast_sample

        probs = torch.softmax(output, dim=-1)
        sampled_probs, sampled_tokens = fast_sample(probs, num_samples=1)
        return probs, sampled_probs, sampled_tokens

    def replay(
        self,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        hashes: torch.Tensor,
        apply_mask: torch.Tensor,
        enabled: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_bs = temperatures.shape[0]
        rows = logits.shape[0]
        if raw_bs != self.batch_size:
            self.logits.zero_()
            self.temperatures.fill_(1)
            self.hashes.zero_()
            self.apply_mask.zero_()
            self.enabled.zero_()
        torch._foreach_copy_(
            [self.logits[:rows], self.temperatures[:raw_bs]],
            [logits, temperatures],
        )
        self.hashes[:rows].copy_(hashes)
        torch._foreach_copy_(
            [self.apply_mask[:rows], self.enabled[:raw_bs]], [apply_mask, enabled]
        )
        self.graph.replay()
        if isinstance(self.output, tuple):
            probs, sampled_probs, sampled_tokens = self.output
            return probs[:rows], sampled_probs[:rows], sampled_tokens[:rows]
        return self.output[:rows]


class MixedTargetGraphs:
    def __init__(
        self,
        *,
        processor: SynthIDWatermarkProcessor,
        capture_bs: list[int],
        rows_per_request: int,
        vocab_size: int,
        sample_draft: bool = False,
    ):
        self.batch_sizes = tuple(sorted(set(capture_bs)))
        stream = torch.cuda.Stream(device=processor.keys.device)
        pool = torch.cuda.graph_pool_handle()
        self.entries = {
            bs: _MixedTargetGraph(
                processor=processor,
                batch_size=bs,
                rows_per_request=rows_per_request,
                vocab_size=vocab_size,
                stream=stream,
                pool=pool,
                sample_draft=sample_draft,
            )
            for bs in self.batch_sizes
        }

    def apply(
        self,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        hashes: torch.Tensor,
        apply_mask: torch.Tensor,
        enabled: torch.Tensor,
    ) -> bool:
        index = bisect_left(self.batch_sizes, temperatures.shape[0])
        if index == len(self.batch_sizes):
            return False
        output = self.entries[self.batch_sizes[index]].replay(
            logits=logits,
            temperatures=temperatures,
            hashes=hashes,
            apply_mask=apply_mask,
            enabled=enabled,
        )
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Target transform graph returned draft samples")
        logits.copy_(output)
        return True

    def sample(
        self,
        *,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        hashes: torch.Tensor,
        apply_mask: torch.Tensor,
        enabled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        index = bisect_left(self.batch_sizes, temperatures.shape[0])
        if index == len(self.batch_sizes):
            return None
        output = self.entries[self.batch_sizes[index]].replay(
            logits=logits,
            temperatures=temperatures,
            hashes=hashes,
            apply_mask=apply_mask,
            enabled=enabled,
        )
        if not isinstance(output, tuple):
            raise RuntimeError("Draft sample graph returned only target logits")
        return output
