"""Finalize statically deferred logits using the current request's policy."""

import torch

from sglang.srt.environ import envs


def make_verify_softcap_ranks(device, softcap):
    """Build an immutable comparison table using the actual device softcap.

    This is model-owned warmup data, not a process-global cache. The table
    represents exact FP32 output ordering/equality for all BF16 bit patterns;
    it does not approximate or change logits. Unsanitized NaNs sort first.
    """
    from sglang.kernels.ops.activation.softcap import softcap_inplace_logits

    raw = (
        torch.arange(65536, device=device, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    values = raw.float()
    if softcap:
        softcap_inplace_logits(values, softcap)
    bits = values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ranks = torch.where(
        (bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
    )
    # Argmax treats signed zeros as equal and selects the first NaN.
    ranks = torch.where(values == 0, 0x80000000, ranks)
    ranks = torch.where(torch.isnan(values), 0xFFFFFFFF, ranks)
    return ranks.to(torch.int32).contiguous()


def can_reduce_verify_logits(batch, grammar_mask=None):
    """Only unmodified greedy verification may discard the full vocabulary."""
    info = batch.sampling_info
    if (
        batch.return_logprob
        or not info.is_all_greedy
        or batch.has_grammar
        or grammar_mask is not None
        or info.has_custom_logit_processor
        or info.has_watermark_candidates
        or envs.SGLANG_SANITIZE_NAN_LOGITS.get()
        or envs.SGLANG_ENABLE_NAN_LOGITS_CHECK.get()
        or envs.SGLANG_ENABLE_ASYNC_ASSERT.get()
    ):
        return False
    if any(
        getattr(info, field, None) is not None
        for field in (
            "acc_additive_penalties",
            "acc_scaling_penalties",
            "logit_bias",
            "grammar_mask",
            "sampling_mask_batch_indices",
            "sampling_support_logprobs_capture_indices",
        )
    ):
        return False
    penalizer = info.penalizer_orchestrator
    return penalizer is None or not penalizer.is_required


def finalize_verify_logits(logits_output, batch, grammar_mask=None):
    """Return compact greedy IDs, or restore ordinary FP32 logits for fallback.

    This runs after replay. Captured graphs never bake request eligibility into
    their output contract.
    """
    raw = getattr(logits_output, "deferred_verify_logits", None)
    if raw is None:
        return None
    if can_reduce_verify_logits(batch, grammar_mask):
        from sglang.kernels.ops.sampling import softcap_argmax

        return softcap_argmax(
            raw,
            softcap=logits_output.deferred_verify_softcap,
            sanitize=False,
            comparison_ranks=getattr(
                logits_output, "deferred_verify_comparison_ranks", None
            ),
        )

    from sglang.kernels.ops.activation.softcap import softcap_inplace_logits

    logits = raw.float()
    if logits_output.deferred_verify_softcap:
        softcap_inplace_logits(logits, logits_output.deferred_verify_softcap)
    from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan

    maybe_detect_nan(logits, "verify: target model logits")
    maybe_detect_inf(logits, "verify: target model logits")
    logits_output.next_token_logits = logits
    logits_output.deferred_verify_logits = None
    return None
