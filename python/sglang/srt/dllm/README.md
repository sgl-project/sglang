# DiffusionGemma output logprobs

Gemma4Renoise supports `return_logprob=true` and optional `token_ids_logprob`
through `/generate`. Scores appear in the existing `meta_info.output_token_logprobs`
and `meta_info.output_token_ids_logprobs` fields, with one row per output token.
Use token IDs from the model's tokenizer, for example the single-token labels
for `A`, `B`, and `C`:

```python
payload = {
    "input_ids": chat_template_token_ids,
    "sampling_params": {"max_new_tokens": 16},
    "return_logprob": True,
    "logprob_start_len": -1,
    "token_ids_logprob": candidate_token_ids,
}
```

For each output position, the score is `log_softmax(raw_logits)` over the entire
vocabulary at that request's **last active denoising step**. It is read at the
same canvas position as the emitted token, without a causal next-token shift.
It is not temperature-scaled, candidate-normalized, an autoregressive sequence
likelihood, or a calibrated probability of correctness. Enabling logprobs does
not change the renoise schedule or add a model forward.

Read the row at your answer token's position to compare candidate labels. Account
for any generated chat-template prefix (such as an empty thought channel), and
validate that prefix before interpreting a fixed answer slot. Multi-token labels
are not scored as sequences. Candidate order and duplicate IDs are preserved.
To obtain a distribution over the supplied choices, apply softmax to their
returned logprobs. The sum of their exponentiated logprobs instead measures the
vocabulary probability mass assigned to those choices.

Both synchronous and first-done-first-out (`--dllm-fdfo`) scheduling are supported.
Unresolved blocks emit no scores. Finished rows retain their scores even if other
rows continue denoising, and normal output truncation and streaming offsets apply.

This support is output-only: omit `logprob_start_len`, set it to `-1`, or set it
to the prompt length. Prompt logprobs, `top_logprobs_num`, flat raw top logprobs,
sampling masks, and logprobs with `max_new_tokens=0` remain unsupported. The
candidate field requires `return_logprob=true`. No decision endpoint is added.
