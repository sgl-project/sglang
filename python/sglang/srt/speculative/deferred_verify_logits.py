"""Finalize statically deferred logits using the current request's policy."""

from sglang.srt.environ import envs


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
            raw, softcap=logits_output.deferred_verify_softcap, sanitize=False
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
