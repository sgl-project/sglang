---
title: "SynthID Text Watermarking"
description: "Configure per-request SynthID Text watermarking for ordinary and EAGLE or EAGLE3 speculative generation."
---

SGLang supports the [SynthID Text](https://github.com/google-deepmind/synthid-text) tournament watermark on NVIDIA CUDA. Watermarking is per request and OFF by default; watermarked and unwatermarked requests share batches.

For the default Aaronson-Gumbel algorithm, and for the settings both algorithms share, see [Text Watermarking](/docs/advanced_features/text_watermarking).

## Server configuration

Store the configuration in a file readable only by the server account. The keys below are public test values, not production secrets:

```json
{
  "keys": [654, 400, 836, 123],
  "ngram_len": 5,
  "num_candidates": 1024,
  "context_history_size": 1024
}
```

```bash
chmod 600 /run/secrets/synthid.json
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --enable-watermark \
  --watermark-algorithm synthid \
  --watermark-config /run/secrets/synthid.json
```

| Field | Default | Meaning |
| --- | --- | --- |
| `keys` | Required | 1 to 64 distinct signed 64-bit integer layer keys. |
| `ngram_len` | `5` | Hash-context n-gram length, from 2 to 17. |
| `num_candidates` | `1024` | Candidate cap of a watermarked row, from 2 to 65,536, clamped to the vocabulary size. |
| `context_history_size` | `1024` | Number of recent context hashes kept per request, from 1 to 2,048; a repeated context is not watermarked again. |
| `skip_first_ngram_calls` | `false` | Do not watermark the first `ngram_len - 1` generated tokens. |

`default_enabled` and `enforce_all` work as for Gumbel; see [Server configuration](/docs/advanced_features/text_watermarking#server-configuration). Fields of the other algorithm are rejected.

## Request control

Requests turn watermarking on and off with the `watermark` object described in [Per-request control](/docs/advanced_features/text_watermarking#per-request-control); with `synthid` only its `enabled` field is accepted. Greedy requests (`top_k` 1 or temperature 0) are never watermarked.

## Sampling

A watermarked row is sampled as follows:

1. Penalties, logit bias, and the grammar mask apply as usual.
2. The row keeps its `num_candidates` highest logits, applies its temperature, then its request top-k, top-p, and min-p.
3. The tournament reweights the remaining candidates by their g-values for the row's context.
4. The stock sampler draws from the result.

Unwatermarked rows in the same batch draw the same tokens as on a server without watermarking. Returned logprobs describe the logits before the watermark; an unwatermarked row's match a server without watermarking up to float rounding (about 1e-7). `SGLANG_RETURN_ORIGINAL_LOGPROB` keeps its meaning. A returned sampling mask describes the watermarked support the token was drawn from.

## Speculative decoding

SynthID supports EAGLE and EAGLE3 with `--speculative-eagle-topk 1`, `--speculative-use-rejection-sampling`, and at most 8 `--speculative-num-draft-tokens`:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-use-rejection-sampling \
  --enable-watermark \
  --watermark-algorithm synthid \
  --watermark-config /run/secrets/synthid.json
```

The watermark applies to the target distribution at verification; the draft model proposes as usual. Rejection sampling then emits tokens from the watermarked target distribution, and only accepted tokens, including the bonus token, advance a request's watermark state.

## Unsupported configurations

`--watermark-algorithm synthid` is rejected with:

- `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`;
- `--rl-on-policy-target`, since the watermark changes the sampling distribution.

The restrictions of `--enable-watermark` that hold for every algorithm, such as PD disaggregation and custom logit processors, are listed under [Compatibility](/docs/advanced_features/text_watermarking#compatibility).

## State and detection

A request's context starts empty (zero tokens), not from the prompt, and holds its last `ngram_len - 1` generated tokens. A request resumed after retraction replays its generated tokens, so its watermark continues as if uninterrupted. The repeated-context history is a bounded ring: a context can be watermarked again after it is evicted.

To score text, recompute each generated token's g-values with the same keys, `ngram_len`, `context_history_size`, and `skip_first_ngram_calls`, starting from the same empty context. Calibrate detection thresholds for the output length and the false-positive rate required; a low score is not proof of authorship.
