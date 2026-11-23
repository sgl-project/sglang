"""TOP admission and accept rule of DeepSeek-V4.1's DSpark draft (its bundled 3-stage block drafter).

Verify scores the anchor and the block's drafts as one static chain on the target's TOP program; each verify row's
target token is drawn with TOP's exponential race keyed by the request's seed and the row's input position (the key
non-speculative decode uses for that token), and drafts are accepted while they equal those targets. Every emitted
token is the one non-speculative TOP decode would draw, whatever DSpark proposes; DSpark's rejection-sampling accept
(statistically exact, but other tokens) is not used.

Since targets do not depend on drafts, DSpark is free to choose its proposals: each Markov step draws its draft with the
race over the draft distribution under the key of the verify row it is matched to (race_draft). Shared noise couples
draft and target, so they agree far more often than independent draws would; emitted tokens are unchanged.
"""

# (steps, top-k, draft tokens) SGLang resolves for DSpark at block size 5: one step, one chain, anchor + 5 drafts.
DSPARK_RECIPE = (1, 1, 6)
DSPARK_BLOCK_SIZE = 5


def validate_dspark(config, args, model_path):
    """Admit DSpark at block size 5 with static verify and the target's own draft; refuse every other form.

    ``args`` are the resolved speculative settings (runtime_context.get_spec())."""
    if args.speculative_algorithm is None:
        return
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    if str(args.speculative_algorithm).upper() != "DSPARK":
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 speculates only with its bundled DSpark draft"
        )
    recipe = (
        args.speculative_num_steps,
        args.speculative_eagle_topk,
        args.speculative_num_draft_tokens,
    )
    if (
        recipe != DSPARK_RECIPE
        or args.speculative_dspark_block_size != DSPARK_BLOCK_SIZE
    ):
        raise NotImplementedError(
            f"[top] DeepSeek-V4.1 DSpark is qualified at block size {DSPARK_BLOCK_SIZE}, recipe {DSPARK_RECIPE}; "
            f"got block size {args.speculative_dspark_block_size}, recipe {recipe}"
        )
    if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 DSpark verifies a static chain (the TOP attention program keys verify rows by a "
            "fixed draft count); ragged and compact verify are refused"
        )
    if args.speculative_use_rejection_sampling:
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 DSpark accepts by TOP's race, not rejection sampling"
        )
    if args.speculative_use_block_verification:
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 DSpark accepts drafts one by one against the race targets"
        )
    if args.speculative_draft_model_path not in (None, model_path):
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 DSpark drafts with the checkpoint's own stages"
        )
    if config.num_nextn_predict_layers != 3:
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 DSpark requires the checkpoint's three draft stages"
        )


def race_accept(
    *, candidates, target_logits, sampling_info, positions, verify_num_draft_tokens
):
    """(accepted draft count, bonus token, zero cap trim) per request under TOP's race.

    ``candidates`` [bs, verify] is the anchor then the drafts; ``positions`` [bs * verify] the verify rows' input
    positions."""
    import torch

    from sglang.kernels.ops.speculative.dflash import (
        compute_dflash_correct_drafts_and_bonus,
    )
    from sglang.srt.speculative.eagle_utils import _top_race_verify_targets

    targets = _top_race_verify_targets(
        next_token_logits=target_logits,
        temperatures=sampling_info.temperatures,
        sampling_seed=sampling_info.sampling_seed,
        positions=positions,
        draft_token_num=verify_num_draft_tokens,
    )
    correct_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates, target_predict=targets
    )
    return correct_len, bonus, torch.zeros_like(correct_len)


def draft_races() -> bool:
    """Whether DSpark draws its drafts with the verify race's keys: the TOP sampler is on."""
    from sglang.srt.runtime_context import get_exec
    from sglang.srt.true_on_policy.config import is_true_on_policy_enabled

    return (
        is_true_on_policy_enabled()
        and get_exec().deterministic.enable_deterministic_inference
    )


def race_draft(*, step_logits, step_temperatures, argmax_tokens, seeds, positions):
    """Stands in for DSpark's Markov draft sampling (MarkovSampleStep) for acceptance, not exactness: a draft raced
    under the key of the verify row it must equal agrees with TOP's race target far more often; emitted tokens are
    unchanged.

    Draft token [bs] of one Markov step, raced under the key of the verify row whose target it must equal.

    Draft step j proposes the token after verify row j (the anchor is row 0), so ``positions`` is that row's input
    position. ``step_logits`` [bs, vocab] are the step's draft logits; rows at temperature 0 keep ``argmax_tokens``.
    Graph-safe: no host reads."""
    import torch

    from sglang.srt.true_on_policy.exponential_race import sample_from_logprobs

    sampled = step_temperatures > 0
    scaled = step_logits.float() / torch.where(sampled, step_temperatures, 1.0)[:, None]
    raced = sample_from_logprobs(scaled, seeds, positions).view(-1)
    return torch.where(sampled, raced, argmax_tokens)
