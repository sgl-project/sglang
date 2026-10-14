"""Draft proposal distribution q for speculative decoding.

``--speculative-draft-temperature``, ``--speculative-draft-top-k`` and
``--speculative-draft-top-p`` give the draft its own sampling policy. Unset,
the draft samples at the request's temperature with no cutoff, so the default
path runs no top-k/top-p kernel. Rejection sampling is unbiased for any q as
long as verification uses the exact q a proposal was drawn from, so callers
keep the q built here or rebuild it from the same logits and params.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import torch.nn.functional as F

from sglang.kernels.ops.sampling import softmax as sampling_softmax
from sglang.kernels.ops.sampling import top_k_renorm_probs, top_p_renorm_probs
from sglang.kernels.ops.speculative.dflash import selector_walk_triton
from sglang.srt.sampling.sampling_params import TOP_K_ALL


class DraftOverrides(msgspec.Struct, frozen=True):
    """The server's draft sampling options; a disabling value (top-k ``-1``,
    top-p ``1``) reads as unset."""

    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]

    @classmethod
    def from_spec(cls) -> DraftOverrides:
        from sglang.srt.runtime_context import get_spec

        try:
            spec = get_spec()
        except ValueError:
            # Unit tests drive the draft paths without a published spec.
            return cls(temperature=None, top_k=None, top_p=None)
        top_k = spec.speculative_draft_top_k
        top_p = spec.speculative_draft_top_p
        return cls(
            temperature=spec.speculative_draft_temperature,
            top_k=None if top_k is None or top_k == -1 else min(top_k, TOP_K_ALL),
            top_p=None if top_p is None or top_p == 1 else top_p,
        )


class DraftSamplingParams(msgspec.Struct, frozen=True):
    """Per-request draft temperature, top-k and top-p, each shaped [bs];
    ``top_ks == 1`` marks a greedy row. The apply flags mirror the server's
    static overrides: unset, no renorm kernel runs."""

    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    apply_top_k: bool = False
    apply_top_p: bool = False

    @property
    def greedy_mask(self) -> torch.Tensor:
        return self.top_ks <= 1

    @classmethod
    def greedy(
        cls,
        batch_size: int,
        device,
        *,
        apply_top_k: bool = False,
        apply_top_p: bool = False,
    ) -> DraftSamplingParams:
        return cls(
            temperatures=torch.ones(batch_size, dtype=torch.float32, device=device),
            top_ks=torch.ones(batch_size, dtype=torch.int32, device=device),
            top_ps=torch.ones(batch_size, dtype=torch.float32, device=device),
            apply_top_k=apply_top_k,
            apply_top_p=apply_top_p,
        )

    @classmethod
    def buffers(cls, max_bs: int, device) -> DraftSamplingParams:
        """Static graph buffers; ``copy_from`` stages each batch into them."""
        overrides = DraftOverrides.from_spec()
        return cls.greedy(
            max_bs,
            device,
            apply_top_k=overrides.top_k is not None,
            apply_top_p=overrides.top_p is not None,
        )

    @classmethod
    def from_sampling_info(
        cls, sampling_info, batch_size: int | None = None, device=None
    ) -> DraftSamplingParams:
        """The request's temperature and greedy rows with each set override in
        place; the request's own top-k/top-p never reach the draft. Graph-safe:
        no host reads of per-request values."""
        overrides = DraftOverrides.from_spec()
        apply_top_k = overrides.top_k is not None
        apply_top_p = overrides.top_p is not None
        if sampling_info is None:
            return cls.greedy(
                batch_size, device, apply_top_k=apply_top_k, apply_top_p=apply_top_p
            )
        bs = len(sampling_info.top_ks) if batch_size is None else batch_size
        temperatures = sampling_info.temperatures.reshape(-1)[:bs].float()
        greedy = sampling_info.top_ks.reshape(-1)[:bs] <= 1
        if overrides.temperature == 0:
            greedy = torch.ones_like(greedy)
        elif overrides.temperature is not None:
            temperatures = torch.full_like(temperatures, overrides.temperature)
        top_k = TOP_K_ALL if overrides.top_k is None else overrides.top_k
        top_ks = torch.full_like(greedy, top_k, dtype=torch.int32)
        top_ks.masked_fill_(greedy, 1)
        top_p = 1.0 if overrides.top_p is None else overrides.top_p
        top_ps = torch.full_like(temperatures, top_p)
        return cls(
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            apply_top_k=apply_top_k,
            apply_top_p=apply_top_p,
        )

    def slice(self, batch_size: int) -> DraftSamplingParams:
        return DraftSamplingParams(
            temperatures=self.temperatures[:batch_size],
            top_ks=self.top_ks[:batch_size],
            top_ps=self.top_ps[:batch_size],
            apply_top_k=self.apply_top_k,
            apply_top_p=self.apply_top_p,
        )

    def copy_from(self, sampling_info, bs: int) -> None:
        """Stage a batch into these graph buffers in place. Padding rows keep
        earlier, still valid values; their outputs are discarded."""
        params = self.from_sampling_info(
            sampling_info, batch_size=bs, device=self.top_ks.device
        )
        assert (params.apply_top_k, params.apply_top_p) == (
            self.apply_top_k,
            self.apply_top_p,
        ), "draft sampling buffers predate the server's draft cutoff options"
        self.temperatures[:bs].copy_(params.temperatures)
        self.top_ks[:bs].copy_(params.top_ks)
        self.top_ps[:bs].copy_(params.top_ps)


def build_draft_probs(
    logits: torch.Tensor, params: DraftSamplingParams
) -> torch.Tensor:
    """q = softmax(logits / T) over the last dim, cut by top-k then top-p where
    the server sets a draft override. ``logits`` is [bs, ..., support]; all
    rows of a request share its params. Greedy rows are the caller's: only
    candidate_probs needs them as point masses."""
    if logits.shape[0] != params.top_ks.shape[0]:
        raise ValueError("Draft logits and sampling params batch sizes differ")
    if logits.numel() == 0:
        return torch.empty_like(logits, dtype=torch.float32)
    flat = logits.reshape(-1, logits.shape[-1]).float()
    rows = flat.shape[0] // logits.shape[0]
    temperatures, top_ks, top_ps = params.temperatures, params.top_ks, params.top_ps
    if rows != 1:
        temperatures, top_ks, top_ps = (
            x.repeat_interleave(rows) for x in (temperatures, top_ks, top_ps)
        )
    probs = sampling_softmax(flat, temperatures=temperatures[:, None])
    if params.apply_top_k:
        probs = top_k_renorm_probs(probs, top_ks)
    if params.apply_top_p:
        probs = top_p_renorm_probs(probs, top_ps)
    return probs.view(logits.shape)


def candidate_probs(
    scores: torch.Tensor, params: Optional[DraftSamplingParams]
) -> torch.Tensor:
    """q for every conditional row of a [bs, slots, K, K] candidate lattice;
    ``params=None`` walks greedily. Greedy rows are point masses: here q alone
    drives min(1, p/q). Draft cutoffs narrow the K candidates, never add."""
    one_hot = F.one_hot(scores.argmax(dim=-1), scores.shape[-1]).float()
    if params is None:
        return one_hot
    probs = build_draft_probs(scores, params)
    return torch.where(params.greedy_mask.view(-1, 1, 1, 1), one_hot, probs)


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
