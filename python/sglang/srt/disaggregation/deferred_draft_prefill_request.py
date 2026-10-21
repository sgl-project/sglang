from __future__ import annotations


def validate_deferred_draft_prefill_request(
    req, tokenized_req, max_input_tokens: int
) -> str | None:
    """Reject unsupported requests before grammar, KV, or hidden admission."""
    if not 0 < len(req.origin_input_ids) <= max_input_tokens:
        return f"Deferred draft prefill requires 1..{max_input_tokens} input tokens"
    if getattr(tokenized_req, "mm_inputs", None) is not None:
        return "Deferred draft prefill currently accepts text prompts only"
    if getattr(tokenized_req, "input_embeds", None) is not None:
        return "Deferred draft prefill does not support input embeddings"
    if any(
        getattr(req.sampling_params, name, None) is not None
        for name in ("json_schema", "regex", "ebnf", "structural_tag")
    ):
        return "Deferred draft prefill does not support constrained decoding"
    if (
        getattr(tokenized_req, "session_params", None) is not None
        or getattr(tokenized_req, "session_id", None) is not None
    ):
        return "Deferred draft prefill does not support sessions"
    if getattr(tokenized_req, "lora_id", None) is not None:
        return "Deferred draft prefill does not support LoRA"
    if getattr(tokenized_req, "custom_logit_processor", None) is not None:
        return "Deferred draft prefill does not support custom logit processors"
    return None
