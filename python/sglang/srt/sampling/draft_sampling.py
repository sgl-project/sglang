"""Draft proposal distribution q for speculative decoding.

Each ``--speculative-draft-*`` option, when set, replaces the request's value
for drafting only. Rejection sampling is unbiased only if verification uses the
exact q a proposal was drawn from, so callers keep the q built here.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import torch.nn.functional as F

from sglang.kernels.ops.sampling import softmax as sampling_softmax
from sglang.kernels.ops.speculative.dflash import selector_walk_triton
from sglang.srt.sampling.probability_transforms import (
    top_k_renorm_probs,
    top_p_renorm_probs,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL


class DraftSamplingParams(msgspec.Struct, frozen=True):
    """Per-request draft temperature, top-k and top-p, each shaped [bs].

    Greedy rows have ``top_k == 1``, as in SamplingParams, so temperatures
    stay positive.
    """

    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor

    @property
    def greedy_mask(self) -> torch.Tensor:
        return self.top_ks <= 1

    @classmethod
    def greedy(cls, batch_size: int, device) -> DraftSamplingParams:
        return cls(
            temperatures=torch.ones(batch_size, dtype=torch.float32, device=device),
            top_ks=torch.ones(batch_size, dtype=torch.int32, device=device),
            top_ps=torch.ones(batch_size, dtype=torch.float32, device=device),
        )

    @classmethod
    def from_sampling_info(
        cls, sampling_info, batch_size: int | None = None, device=None
    ) -> DraftSamplingParams:
        """Apply the draft overrides to the target's per-request values.

        Greedy target rows stay greedy; draft temperature 0 makes every row
        greedy. Graph-safe: no host reads of per-request values.
        """
        from sglang.srt.runtime_context import get_spec

        if sampling_info is None:
            return cls.greedy(batch_size, device)
        bs = len(sampling_info.top_ks) if batch_size is None else batch_size
        temperatures = sampling_info.temperatures.reshape(-1)[:bs].float()
        top_ks = sampling_info.top_ks[:bs]
        top_ps = sampling_info.top_ps[:bs]
        spec = get_spec()
        if (top_k := spec.speculative_draft_top_k) is not None:
            top_k = TOP_K_ALL if top_k == -1 else min(top_k, TOP_K_ALL)
            top_ks = top_ks.masked_fill(top_ks > 1, top_k)
        if (temperature := spec.speculative_draft_temperature) == 0:
            top_ks = torch.ones_like(top_ks)
        elif temperature is not None:
            temperatures = torch.full_like(temperatures, temperature)
        if (top_p := spec.speculative_draft_top_p) is not None:
            top_ps = torch.full_like(top_ps, top_p)
        return cls(temperatures, top_ks, top_ps)

    def slice(self, batch_size: int) -> DraftSamplingParams:
        return DraftSamplingParams(
            self.temperatures[:batch_size],
            self.top_ks[:batch_size],
            self.top_ps[:batch_size],
        )

    def copy_from(self, sampling_info, bs: int) -> None:
        """Stage a batch into these graph buffers in place. Padding rows keep
        earlier, still valid values; their outputs are discarded."""
        params = self.from_sampling_info(
            sampling_info, batch_size=bs, device=self.top_ks.device
        )
        self.temperatures[:bs].copy_(params.temperatures)
        self.top_ks[:bs].copy_(params.top_ks)
        self.top_ps[:bs].copy_(params.top_ps)


def build_draft_probs(
    logits: torch.Tensor, params: DraftSamplingParams
) -> torch.Tensor:
    """Return q = top_p(top_k(softmax(logits / T))) over the last dim.

    ``logits`` is [bs, ..., support], where the support is the vocabulary or a
    model's candidate set; all rows of a request share its params. Greedy rows
    are a point mass at the argmax, ties included.
    """
    if logits.shape[0] != params.top_ks.shape[0]:
        raise ValueError("Draft logits and sampling params batch sizes differ")
    if logits.numel() == 0:
        return torch.empty_like(logits, dtype=torch.float32)
    flat = logits.reshape(-1, logits.shape[-1]).float()
    rows = flat.shape[0] // logits.shape[0]
    temperatures, top_ks, top_ps = (
        x.repeat_interleave(rows)
        for x in (params.temperatures, params.top_ks, params.top_ps)
    )
    probs = sampling_softmax(flat, temperatures=temperatures[:, None])
    probs = top_p_renorm_probs(top_k_renorm_probs(probs, top_ks), top_ps)
    greedy = (top_ks <= 1)[:, None]
    probs.masked_fill_(greedy, 0.0).scatter_add_(
        -1, flat.argmax(dim=-1, keepdim=True), greedy.float()
    )
    return probs.view(logits.shape)


def candidate_probs(
    scores: torch.Tensor, params: Optional[DraftSamplingParams]
) -> torch.Tensor:
    """q for every conditional row of a [bs, slots, K, K] candidate lattice.

    ``params=None`` walks greedily: each row is a point mass at its argmax.
    Draft top-k/top-p narrow the K candidates; they never add tokens.
    """
    if params is None:
        return F.one_hot(scores.argmax(dim=-1), scores.shape[-1]).float()
    return build_draft_probs(scores, params)


def sample_indices_from_probs(
    probs: torch.Tensor, uniforms: torch.Tensor
) -> torch.Tensor:
    indices = uniforms.ge(probs.cumsum(dim=-1)).sum(dim=-1)
    offsets = torch.arange(probs.shape[-1], device=probs.device)
    # Roundoff can leave the CDF just below one; fall back inside q's support.
    last_supported = torch.where(probs > 0, offsets, 0).amax(dim=-1)
    return torch.minimum(indices, last_supported)


def sample_candidate_path(
    *, candidate_ids: torch.Tensor, probs: torch.Tensor, uniforms: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Walk a [bs, slots, K, K] lattice: each slot samples from the q row its
    predecessor picked. Returns the tokens and the realized q rows."""
    if probs.is_cuda:
        return selector_walk_triton(
            candidate_ids=candidate_ids, probs=probs, uniforms=uniforms
        )
    batch, num_slots, topk = candidate_ids.shape
    previous = torch.zeros(batch, dtype=torch.int64, device=probs.device)
    tokens, q_rows = [], []
    for slot in range(num_slots):
        q_row = probs[:, slot].gather(1, previous.view(-1, 1, 1).expand(-1, 1, topk))
        q_row = q_row.squeeze(1)
        previous = sample_indices_from_probs(q_row, uniforms[:, slot : slot + 1])
        q_rows.append(q_row)
        tokens.append(candidate_ids[:, slot].gather(1, previous.view(-1, 1)).squeeze(1))
    return torch.stack(tokens, dim=-1).to(torch.int64), torch.stack(q_rows, dim=1)
