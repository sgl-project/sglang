"""TOP admission of EAGLE speculative decoding with the checkpoint's own MTP layer as the draft.

Rollout log-probs come from the target model, so every token stays trainer-exact with speculation on.
"""


def validate_mtp(config, args, *, family, extra_recipes=()):
    """Admit the qualified single-proposal and FP8 depth-five recipes (the checkpoint MTP layer drafts);
    ``extra_recipes`` are the family's additional qualified (steps, topk, draft tokens)."""
    if not args.speculative_algorithm:
        return
    from sglang.srt.true_on_policy import fp8

    if args.speculative_algorithm != "EAGLE":
        raise NotImplementedError(
            f"[top] {family} MTP requires speculative_algorithm='EAGLE'"
        )
    recipe = (
        args.speculative_num_steps,
        args.speculative_eagle_topk,
        args.speculative_num_draft_tokens,
    )
    allowed_recipes = ((1, 1, 2), (5, 1, 6)) if fp8.enabled() else ((1, 1, 2),)
    allowed_recipes += tuple(extra_recipes)
    if recipe not in allowed_recipes:
        raise NotImplementedError(
            f"[top] {family} MTP requires (steps, topk, draft tokens) in {allowed_recipes}"
        )
    if getattr(config, "num_nextn_predict_layers", 0) != 1:
        raise NotImplementedError(
            f"[top] {family} MTP requires one checkpoint MTP layer"
        )
    if args.speculative_draft_model_path != args.model_path:
        raise NotImplementedError(f"[top] {family} MTP requires the target checkpoint")
    allowed_quant = (None, "fp8") if fp8.enabled() else (None,)
    if args.speculative_draft_model_quantization not in allowed_quant:
        raise NotImplementedError(
            f"[top] {family} MTP draft quantization must match the admitted checkpoint"
        )
    for draft, target in (
        ("speculative_moe_a2a_backend", "moe_a2a_backend"),
        ("speculative_moe_runner_backend", "moe_runner_backend"),
        ("speculative_draft_attention_backend", "attention_backend"),
    ):
        value = getattr(args, draft)
        if value is not None and value != getattr(args, target):
            raise NotImplementedError(
                f"[top] {family} MTP requires {draft} to match {target}"
            )
