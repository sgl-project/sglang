from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from flashinfer import top_k as _flashinfer_top_k

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    build_dflash_verify_target_probs,
)
from sglang.srt.speculative.spec_utils import fast_sample

_SPARSE_TOP_K_LIMIT = 128


def _normalize_sparse_topk_probs(
    topk_logits: torch.Tensor,
    temperatures: torch.Tensor,
    valid: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Normalize a compact top-k support with top-k-first top-p semantics."""
    scaled = topk_logits.float() / temperatures
    scaled = scaled.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    probs = probs.masked_fill((cdf - probs) > top_ps, 0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.compile(dynamic=True)
def _sparse_rejection_from_support(
    candidates: torch.Tensor,
    target_ids: torch.Tensor,
    target_probs: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    accept_uniforms: torch.Tensor,
    final_uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact p/q rejection sampling on compact linear-chain supports."""
    batch_size, forward_width = candidates.shape
    num_proposals = forward_width - 1

    if num_proposals == 0:
        final_ids = target_ids[:, 0]
        final_probs = target_probs[:, 0]
        accepted_counts = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=candidates.device,
        )
    else:
        proposal_ids = candidates[:, 1:]
        p_proposal = torch.where(
            target_ids[:, :num_proposals].eq(proposal_ids.unsqueeze(-1)),
            target_probs[:, :num_proposals],
            torch.zeros(
                (),
                dtype=target_probs.dtype,
                device=target_probs.device,
            ),
        ).sum(dim=-1)
        q_proposal = torch.where(
            draft_ids.eq(proposal_ids.unsqueeze(-1)),
            draft_probs,
            torch.zeros(
                (),
                dtype=draft_probs.dtype,
                device=draft_probs.device,
            ),
        ).sum(dim=-1)
        ratios = torch.where(
            q_proposal > 0,
            p_proposal / q_proposal,
            torch.zeros_like(p_proposal),
        ).clamp_(max=1.0)
        accepted_flags = accept_uniforms[:, :num_proposals] < ratios
        accepted_counts = accepted_flags.to(torch.int32).cumprod(dim=1).sum(dim=1)

        batch_indices = torch.arange(batch_size, device=candidates.device)
        final_rows = accepted_counts.to(torch.long)
        final_ids = target_ids[batch_indices, final_rows]
        final_probs = target_probs[batch_indices, final_rows]

        rejected = final_rows < num_proposals
        draft_rows = final_rows.clamp(max=num_proposals - 1)
        final_draft_ids = draft_ids[batch_indices, draft_rows]
        final_draft_probs = draft_probs[batch_indices, draft_rows]
        q_on_target = torch.where(
            final_ids.unsqueeze(2).eq(final_draft_ids.unsqueeze(1)),
            final_draft_probs.unsqueeze(1),
            torch.zeros(
                (),
                dtype=final_draft_probs.dtype,
                device=final_draft_probs.device,
            ),
        ).sum(dim=2)
        correction_probs = (final_probs - q_on_target).clamp_min_(0.0)
        correction_sum = correction_probs.sum(dim=1, keepdim=True)
        correction_probs = torch.where(
            correction_sum > 0,
            correction_probs / correction_sum.clamp_min(1e-12),
            final_probs,
        )
        final_probs = torch.where(
            rejected[:, None],
            correction_probs,
            final_probs,
        )

    cdf = torch.cumsum(final_probs, dim=-1)
    thresholds = final_uniforms * final_probs.sum(dim=-1)
    sampled_offsets = (cdf <= thresholds[:, None]).sum(dim=-1)
    sampled_offsets.clamp_(max=final_ids.shape[-1] - 1)
    bonus = (
        final_ids.gather(1, sampled_offsets[:, None]).squeeze(1).to(candidates.dtype)
    )
    return accepted_counts, bonus


@torch.compile(dynamic=True)
def _build_sparse_target_support_tensors(
    next_token_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    batch_size: int,
    forward_width: int,
    max_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build compact support using the fastest available top-k primitive."""
    rows = batch_size * forward_width
    topk_logits, topk_ids = _flashinfer_top_k(
        next_token_logits.contiguous(),
        max_top_k,
        sorted=True,
        deterministic=False,
    )

    expanded_temperatures = torch.repeat_interleave(
        temperatures,
        forward_width,
        dim=0,
    ).reshape(rows, -1)
    expanded_top_ks = torch.repeat_interleave(
        top_ks,
        forward_width,
        dim=0,
    ).reshape(rows, 1)
    expanded_top_ps = torch.repeat_interleave(
        top_ps,
        forward_width,
        dim=0,
    ).reshape(rows, 1)
    ranks = torch.arange(
        max_top_k,
        dtype=expanded_top_ks.dtype,
        device=next_token_logits.device,
    )[None, :]
    probs = _normalize_sparse_topk_probs(
        topk_logits,
        expanded_temperatures,
        ranks < expanded_top_ks,
        expanded_top_ps,
    )
    return (
        topk_ids.view(batch_size, forward_width, max_top_k),
        probs.view(batch_size, forward_width, max_top_k),
    )


def _build_sparse_target_support(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    batch_size: int,
    forward_width: int,
    max_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact target token IDs/probabilities without a dense scatter."""
    if bool(getattr(sampling_info, "need_top_p_sampling", False)):
        top_ps = sampling_info.top_ps
    else:
        top_ps = torch.ones(
            (batch_size,),
            dtype=torch.float32,
            device=next_token_logits.device,
        )

    return _build_sparse_target_support_tensors(
        next_token_logits,
        sampling_info.temperatures,
        sampling_info.top_ks,
        top_ps,
        batch_size,
        forward_width,
        max_top_k,
    )


def _sample_from_support(
    support_ids: torch.Tensor,
    support_probs: torch.Tensor,
) -> torch.Tensor:
    """Sample one token from every compact support row."""
    flat_ids = support_ids.flatten(0, 1)
    flat_probs = support_probs.flatten(0, 1)
    _, offsets = fast_sample(flat_probs)
    return flat_ids.gather(1, offsets).view(support_ids.shape[:2])


@dataclass(frozen=True)
class UnoDraftDistribution:
    """The exact q used to sample future UNO proposal rows."""

    probs: torch.Tensor
    token_ids: torch.Tensor | None = None


def _run_sparse_rejection(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: int,
    draft_distribution: UnoDraftDistribution,
) -> tuple[torch.Tensor, torch.Tensor]:
    if draft_distribution.token_ids is None:
        raise RuntimeError("Sparse UNO verification requires sparse draft q.")

    batch_size, forward_width = candidates.shape
    support_ids, support_probs = _build_sparse_target_support(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        batch_size=batch_size,
        forward_width=forward_width,
        max_top_k=max_top_k,
    )

    # Preserve the legacy path's two RNG draws and tensor shapes. The final
    # uniforms select the correction/bonus; the first F - 1 acceptance coins
    # are consumed by a linear chain.
    accept_uniforms = torch.rand(
        (batch_size, forward_width),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    final_uniforms = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    return _sparse_rejection_from_support(
        candidates,
        support_ids,
        support_probs,
        draft_distribution.token_ids,
        draft_distribution.probs,
        accept_uniforms,
        final_uniforms,
    )


def _build_dense_probs(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    batch_size: int,
    forward_width: int,
    max_top_k: int,
    uniform_top_k_value: int | None,
) -> torch.Tensor:
    """Build the dense sampling distribution used by SGLang verification."""
    return build_dflash_verify_target_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_token_num=forward_width,
        bs=batch_size,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
        use_sparse_topk=True,
    )


def _sample_from_dense_probs(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token from every dense distribution row."""
    _, token_ids = fast_sample(probs.flatten(0, 1))
    return token_ids.view(probs.shape[:2])


def _run_dense_rejection(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: int,
    uniform_top_k_value: int | None,
    draft_distribution: UnoDraftDistribution,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SGLang's fused linear-chain p/q rejection kernel."""
    if draft_distribution.token_ids is not None:
        raise RuntimeError("Dense UNO verification requires dense draft q.")

    batch_size, forward_width = candidates.shape
    target_probs = _build_dense_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        batch_size=batch_size,
        forward_width=forward_width,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
    )
    accept_uniforms = torch.rand(
        (batch_size, forward_width),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    final_uniforms = torch.rand(
        (batch_size,),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accepted_counts,
    ) = _get_or_create_chain_verify_buffers(
        bs=batch_size,
        draft_token_num=forward_width,
        device=next_token_logits.device,
    )
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accepted_counts,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=accept_uniforms,
        uniform_samples_for_final_sampling=final_uniforms,
        target_probs=target_probs,
        draft_probs=draft_distribution.probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )

    rows = torch.arange(batch_size, device=candidates.device)
    bonus_positions = accept_index[
        rows,
        accepted_counts.to(torch.long),
    ].to(torch.long)
    bonus = predicts[bonus_positions].to(candidates.dtype)
    return accepted_counts, bonus


@dataclass
class UnoSamplingResult:
    output_ids: torch.Tensor
    accept_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    next_seed_tokens: torch.Tensor


@dataclass
class UnoTreeSamplingResult:
    output_ids: torch.Tensor
    accept_lens: torch.Tensor


def build_uno_draft_input(
    *,
    seed_tokens: torch.Tensor,
    forward_width: int,
    vocab_size: int,
    noise_tokens: torch.Tensor | None = None,  # for testing
) -> torch.Tensor:
    """Build one ``[seed, uniform noise...]`` row per request.

    Random noise is sampled independently from ``[0, vocab_size)``.

    Supplying ``noise_tokens`` bypasses random generation. This is used
    by deterministic tests and must have shape
    ``(batch_size, forward_width - 1)``.

    ``forward_width == 1`` never generates noise or consumes RNG state.
    """
    seed_tokens = seed_tokens.reshape(-1).to(dtype=torch.int64)
    batch_size = seed_tokens.numel()
    noise_shape = (batch_size, forward_width - 1)

    if forward_width == 1:
        return seed_tokens[:, None]

    if noise_tokens is None:
        noise_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=noise_shape,
            dtype=torch.int64,
            device=seed_tokens.device,
        )
    else:
        noise_tokens = noise_tokens.to(
            device=seed_tokens.device,
            dtype=torch.int64,
        )

    draft_input_ids = seed_tokens.new_empty((batch_size, forward_width))
    draft_input_ids[:, 0].copy_(seed_tokens)
    draft_input_ids[:, 1:].copy_(noise_tokens)

    return draft_input_ids


def sample_uno_candidates(
    *,
    draft_logits: torch.Tensor,  # [B, F, V]
    sampling_info: Any,
    max_top_k: int,
    uniform_top_k_value: int | None = None,
) -> tuple[torch.Tensor, UnoDraftDistribution]:
    """Sample 1 clean token from seed and F-1 draft tokens.
    Depending on max_top_k value, may use sparse representations for efficiency.
    candidates: [B, F] sampled tokens including clean and draft.
    draft_distribution: probabilities of the draft tokens for rejection sampling.
    """
    batch_size, forward_width, vocab_size = draft_logits.shape
    flat_logits = draft_logits.reshape(-1, vocab_size)
    if max_top_k <= _SPARSE_TOP_K_LIMIT:
        support_ids, support_probs = _build_sparse_target_support(
            next_token_logits=flat_logits,
            sampling_info=sampling_info,
            batch_size=batch_size,
            forward_width=forward_width,
            max_top_k=max_top_k,
        )
        candidates = _sample_from_support(support_ids, support_probs)
        draft_distribution = UnoDraftDistribution(
            token_ids=support_ids[:, 1:],
            probs=support_probs[:, 1:],
        )
    else:
        probs = _build_dense_probs(
            next_token_logits=flat_logits,
            sampling_info=sampling_info,
            batch_size=batch_size,
            forward_width=forward_width,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
        )
        candidates = _sample_from_dense_probs(probs)
        draft_distribution = UnoDraftDistribution(probs=probs[:, 1:])
    return candidates, draft_distribution


def sample_uno_clean_root(
    *,
    seed_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    sampling_info: Any,
    max_top_k: int,
    uniform_top_k_value: int | None = None,
) -> torch.Tensor:
    """Sample the clean root using the current UNO sampling path."""
    del seed_tokens
    candidates, _ = sample_uno_candidates(
        draft_logits=draft_logits[:, :1, :].contiguous(),
        sampling_info=sampling_info,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
    )
    return candidates[:, 0]


def sample_uno_tree_target_tokens(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    batch_size: int,
    verify_width: int,
    max_top_k: int,
) -> torch.Tensor:
    """Sample one target token per verify node from compact top-k support."""
    support_ids, support_probs = _build_sparse_target_support(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        batch_size=batch_size,
        forward_width=verify_width,
        max_top_k=max_top_k,
    )
    return _sample_from_support(support_ids, support_probs)


def pack_uno_tree_result(
    *,
    clean_root_tokens: torch.Tensor,
    eagle_predict: torch.Tensor,
    eagle_accept_lens: torch.Tensor,
    draft_width: int,
) -> UnoTreeSamplingResult:
    """Convert an internal EAGLE tree result into UNO's public row."""
    clean_root_tokens = clean_root_tokens.reshape(-1).to(dtype=torch.int64)
    batch_size = clean_root_tokens.numel()
    verify_width = eagle_predict.numel() // batch_size
    predict_rows = eagle_predict.reshape(batch_size, verify_width)

    output_ids = clean_root_tokens.new_zeros((batch_size, draft_width + 1))
    output_ids[:, 0].copy_(clean_root_tokens)
    output_ids[:, 1:].copy_(predict_rows[:, :draft_width].to(dtype=output_ids.dtype))
    return UnoTreeSamplingResult(
        output_ids=output_ids,
        accept_lens=eagle_accept_lens + 1,
    )


def pack_uno_result(
    *,
    candidates: torch.Tensor,  # [B, F]
    accepted_proposal_counts: torch.Tensor,  # [B]
    bonus_tokens: torch.Tensor,  # [B]
    committed_frontiers: torch.Tensor,  # [B]
) -> UnoSamplingResult:
    """Pack acceptance into fixed-width UNO output rows."""
    batch_size, forward_width = candidates.shape
    output_ids = candidates.new_zeros((batch_size, forward_width + 1))
    output_ids[:, :forward_width].copy_(candidates)
    output_ids.scatter_(
        1,
        (accepted_proposal_counts.to(torch.long) + 1)[:, None],
        bonus_tokens[:, None],
    )

    accept_lens = accepted_proposal_counts + 2
    new_seq_lens = committed_frontiers + accept_lens.to(committed_frontiers.dtype)
    return UnoSamplingResult(
        output_ids=output_ids,
        accept_lens=accept_lens,
        new_seq_lens=new_seq_lens,
        next_seed_tokens=bonus_tokens,
    )


def run_uno_sampling(
    *,
    candidates: torch.Tensor,  # [B, F]
    next_token_logits: torch.Tensor,  # [B x F, V]
    sampling_info: Any,
    committed_frontiers: torch.Tensor,  # [B]
    draft_distribution: UnoDraftDistribution,
    max_top_k: int,
    uniform_top_k_value: int | None = None,
) -> UnoSamplingResult:
    """Verify sampled UNO proposals against target p and pack the result."""
    if draft_distribution.token_ids is not None:
        accepted, bonus = _run_sparse_rejection(
            candidates=candidates,
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            max_top_k=max_top_k,
            draft_distribution=draft_distribution,
        )
    else:
        accepted, bonus = _run_dense_rejection(
            candidates=candidates,
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            draft_distribution=draft_distribution,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
        )
    return pack_uno_result(
        candidates=candidates,
        accepted_proposal_counts=accepted,
        bonus_tokens=bonus,
        committed_frontiers=committed_frontiers,
    )
