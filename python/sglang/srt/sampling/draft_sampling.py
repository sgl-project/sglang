"""Sampling policy for model-produced full-vocabulary or candidate logits.

Models own their correction and candidate support. This module transforms the
resulting conditional logits, and callers retain the returned q for verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.kernels.ops.sampling import softmax as sampling_softmax
from sglang.srt.sampling.probability_transforms import (
    top_k_renorm_probs,
    top_p_renorm_probs,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL


@dataclass
class DraftSamplingParams:
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor

    @property
    def greedy_mask(self) -> torch.Tensor:
        return (self.temperatures.reshape(-1) == 0) | (self.top_ks <= 1)

    @classmethod
    def create(cls, batch_size: int, device: torch.device | str) -> DraftSamplingParams:
        return cls(
            temperatures=torch.ones(
                (batch_size, 1), dtype=torch.float32, device=device
            ),
            top_ks=torch.full(
                (batch_size,), TOP_K_ALL, dtype=torch.int32, device=device
            ),
            top_ps=torch.ones((batch_size,), dtype=torch.float32, device=device),
        )

    @classmethod
    def from_sampling_info(
        cls, sampling_info, *, batch_size: int | None = None, device=None
    ) -> DraftSamplingParams:
        from sglang.srt.runtime_context import get_spec

        if sampling_info is None:
            if batch_size is None or device is None:
                raise ValueError(
                    "batch_size and device are required without sampling_info"
                )
            params = cls.create(batch_size, device)
            params.top_ks.fill_(1)
            return params
        bs = len(sampling_info.temperatures) if batch_size is None else batch_size
        spec = get_spec()
        temperatures = sampling_info.temperatures[:bs].reshape(bs, 1)
        top_ks = sampling_info.top_ks[:bs]
        top_ps = sampling_info.top_ps[:bs]
        if spec.speculative_draft_temperature is not None:
            temperatures = torch.full_like(
                temperatures, spec.speculative_draft_temperature
            )
        if spec.speculative_draft_top_k is not None:
            top_k = spec.speculative_draft_top_k
            top_k = TOP_K_ALL if top_k == -1 else min(top_k, TOP_K_ALL)
            # Greedy target rows retain deterministic drafts even when the
            # stochastic proposal cutoff is disabled or widened.
            top_ks = torch.where(top_ks <= 1, 1, torch.full_like(top_ks, top_k))
        if spec.speculative_draft_top_p is not None:
            top_ps = torch.full_like(top_ps, spec.speculative_draft_top_p)
        return cls(temperatures, top_ks, top_ps)

    def slice(self, batch_size: int) -> DraftSamplingParams:
        return DraftSamplingParams(
            self.temperatures[:batch_size],
            self.top_ks[:batch_size],
            self.top_ps[:batch_size],
        )

    def copy_from(self, sampling_info, bs: int) -> None:
        """Stage request values at stable addresses and reset graph padding."""
        params = self.from_sampling_info(
            sampling_info, batch_size=bs, device=self.temperatures.device
        )
        self.temperatures[bs:].fill_(1.0)
        self.top_ks[bs:].fill_(TOP_K_ALL)
        self.top_ps[bs:].fill_(1.0)
        self.temperatures[:bs].copy_(params.temperatures)
        self.top_ks[:bs].copy_(params.top_ks)
        self.top_ps[:bs].copy_(params.top_ps)


def build_draft_probs(
    logits: torch.Tensor, params: DraftSamplingParams
) -> torch.Tensor:
    """Return the actual proposal q for logits shaped [batch, ..., support].

    Apply temperature to the completed model scores, then top-k and top-p in
    that order. The support may be the vocabulary or a model's candidate set.
    No request-dependent host branching is used inside captured graphs.
    """
    if logits.ndim < 2 or logits.shape[0] != params.temperatures.numel():
        raise ValueError(
            "Draft logits and sampling parameters must have the same batch size"
        )
    if logits.numel() == 0:
        return torch.empty_like(logits, dtype=torch.float32)
    batch_size, vocab_size = logits.shape[0], logits.shape[-1]
    rows_per_request = logits.numel() // (batch_size * vocab_size)
    temperatures = params.temperatures.reshape(-1, 1).repeat_interleave(
        rows_per_request, 0
    )
    top_ks = params.top_ks.repeat_interleave(rows_per_request)
    top_ps = params.top_ps.repeat_interleave(rows_per_request)
    greedy = params.greedy_mask.repeat_interleave(rows_per_request)
    flat_logits = logits.reshape(-1, vocab_size).float()
    # Greedy rows are represented by point masses, including when the target
    # remains stochastic. Never divide logits by a zero draft temperature.
    safe_temperatures = torch.where(temperatures == 0, 1.0, temperatures)
    probs = sampling_softmax(flat_logits, temperatures=safe_temperatures)
    probs = top_k_renorm_probs(probs, top_ks)
    probs = top_p_renorm_probs(probs, top_ps)
    argmax = flat_logits.argmax(dim=-1, keepdim=True)
    probs.masked_fill_(greedy[:, None], 0.0)
    probs.scatter_(
        -1, argmax, torch.where(greedy[:, None], 1.0, probs.gather(-1, argmax))
    )
    return probs.reshape(logits.shape)
