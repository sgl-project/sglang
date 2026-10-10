# Text watermarking design

This package implements keyed Aaronson-Gumbel sampling and model-free detection for generated text. Watermarking is opt-in, preserves the target distribution under the ideal hash model, and supports ordinary and speculative decoding. See the [operator guide](../../../../../docs/docs/advanced_features/text_watermarking.mdx) for deployment guidance and the full compatibility matrix.

## Algorithm

After penalties, logit bias, grammar masking, temperature scaling, and top-k/top-p/min-p truncation, let `p[t]` be the probability of token `t`. For each supported token, the selector derives a deterministic uniform `u[t]` from the request key and preceding context, then selects

```text
argmax_t log(u[t]) / p[t]
```

This is the logarithmic form of `argmax_t u[t]^(1 / p[t])`. Independent uniform values select each token with probability `p[t]`; masked and truncated tokens have zero support and cannot win. Ties select the lowest token ID.

All hash arithmetic wraps to unsigned 32 bits. `mix` and `fmix32` are the MurmurHash3 x86 32-bit body mix and finalizer:

```text
mix(state, value):
  value = value * 0xcc9e2d51
  value = rotl32(value, 15)
  value = value * 0x1b873593
  state = rotl32(state XOR value, 13)
  return state * 5 + 0xe6546b64

fmix32(value):
  value = value XOR (value >> 16)
  value = value * 0x85ebca6b
  value = value XOR (value >> 13)
  value = value * 0xc2b2ae35
  return value XOR (value >> 16)
```

A key is 1 to 16 hexadecimal digits with an optional `0x` prefix, interpreted as an unsigned 64-bit integer. For context tokens `c_1 ... c_L`, candidate token `t`, and the low and high 32-bit key words:

```text
context_hash = fmix32(mix(...mix(mix(0, c_1), c_2)..., c_L) XOR (4 * L))
token_hash   = fmix32(mix(mix(mix(mix(0, key_lo), key_hi), context_hash), t) XOR 16)
u            = (token_hash + 0.5) / 2^32
```

The context is the last `h` prompt and committed output tokens before the sampled position. The first generated token therefore uses the prompt tail.

Direct fp32 evaluation of `u` rounds the largest 128 hash values to `1.0`. The Triton and PyTorch selectors preserve relative precision near one by using the direct expression below `2^31` and

```text
log1p(-(2^32 - token_hash - 0.5) / 2^32)
```

above it. The integer complement is formed before conversion to floating point.

## Detection

Detection needs token IDs, the same key and context window, and preferably the exact prompt token IDs. For each generated token with uniform value `u`, it adds `-log(1 - u)` to the score. Under the null hypothesis, the sum over `n` contexts follows `Gamma(shape=n, scale=1)`. The detector reports the one-sided upper-tail p-value, its logarithm, and `z = (score - n) / sqrt(n)`.

The detector scores each exact context tuple once. Runtime uses the 32-bit context hash for admission; a rare collision can only add an ordinary null term to detector scoring and dilute the signal. Without prompt IDs, positions whose initial context is unknown are skipped.

Scoring uses a fixed prefix of the first `min(4,096, per-request token-pool capacity)` distinct contexts. Later positions are ordinary samples after runtime reaches the same history budget, so including them would dilute the test. The model-free detector cannot identify entropy-gated or greedy positions; it scores them as conservative null terms.

The [operator guide's detector specification](../../../../../docs/docs/advanced_features/text_watermarking.mdx#detection) is normative for prompt alignment, prefix handling, dual-key partitioning, and result interpretation.

## Runtime design

- **Policy resolution.** Server policy and each request's `watermark` object resolve to a key, context window, and enabled bit before batch construction. Mixed batches retain independent configuration and state per row.
- **Ring buffer.** Each `req_pool_idx` owns the committed prompt/output tail used to reconstruct contexts. GPU state avoids stale CPU output IDs under overlap scheduling and supplies current contexts for every speculative tree row.
- **Repeated-context masking.** A context hash is forced at most once per request. Repeated contexts use ordinary sampling, preventing a deterministic short cycle from repeatedly excluding EOS. Only forced contexts enter the history.
- **Greedy bypass.** Rows normalized to `top_k <= 1`, including `temperature=0`, use ordinary greedy selection and are not recorded.
- **Entropy gate.** A row uses ordinary sampling when its largest truncated probability exceeds the server's `max_probability`. Skipped contexts remain eligible for a later higher-entropy occurrence.
- **Speculative decoding.** NGRAM, EAGLE, EAGLE3, and NEXTN reconstruct verify-row contexts from the committed ring tail, tree mask, draft tokens, and positions. Normal and speculative decoding force a clone when log probabilities are requested, so logprob computation sees the pre-watermark target distribution. Fused kernels admit contexts and append only accepted tokens.
- **Selector paths.** Finite `top_k <= 8,192` uses `torch.topk`, canonicalizes boundary ties, and performs small-k truncation. Unlimited top-k retains full-vocabulary sorting. Both paths use the same hash and selection contract.
- **Disabled-batch exit.** A host-side candidate bit follows filtering and merging. Fully disabled or greedy-only batches skip prompt initialization, verify cloning, selector work, and append kernels.
- **Retraction and slot reuse.** Re-prefill rebuilds the distinct-context history from accepted output. A new request resets all state for a reused request-pool slot.
- **Dual key.** A deterministic per-context coin chooses key A with probability `P` and key B otherwise:

  ```text
  coin      = fmix32(mix(mix(mix(mix(mix(0, a_lo), a_hi), b_lo), b_hi), context_hash) XOR 20)
  threshold = floor(P * 2^32)
  key       = key A if coin < threshold else key B
  ```

  A request key replaces key A; key B remains server-owned. Omitting key B leaves the single-key path unchanged.

## Package layout

| File | Role |
| --- | --- |
| `core.py` | Request policy resolution, batch configuration, ring/history state, PyTorch reference selection, and speculative orchestration. |
| `config.py` | Key parsing, bounded JSON configuration loading, and shared protocol limits. |
| `detector.py` | Torch-free single-key and dual-key reference detector. |
| `__init__.py` | Minimal public detector exports. |

The Triton selector and state kernels live in `sglang/kernels/ops/sampling/textseal_selector.py`.

## Server configuration

`--enable-watermark` allocates CUDA watermark state and accepts request configuration. All other settings come from one JSON object passed with `--watermark-config`, as a file path or inline JSON (a value starting with `{`); every field is optional and unknown fields are rejected.

| Field | Default | Behavior |
| --- | --- | --- |
| `key` | unset | Server key A. |
| `key_b` | unset | Enables dual-key mode with server key B. Requires key A. |
| `context_window` | `4` | Default context window and per-request maximum, from 1 to 64. |
| `mixing_probability` | `0.5` | Probability of selecting key A for an eligible context; a non-default value requires `key_b`. |
| `max_probability` | `1.0` | Uses ordinary sampling when the largest truncated probability exceeds this value. |
| `default_enabled` | `false` | Uses the server key when a request omits `watermark`; explicit opt-out remains allowed. |
| `enforce_all` | `false` | Uses the server key by default and rejects explicit opt-out. |

Default-on and enforce-all modes require a server key. A server key without either mode remains available only to requests that opt in.

Use a regular config file readable only by the server account for keys; inline JSON is visible in the process command line. Launch-command, request-log, server-info, and crash-payload rendering redacts key fields and the `--watermark-config` value. Process command lines, client request bodies, and process or GPU core dumps remain separate secret-bearing surfaces.

## Per-request control

The native `/generate` endpoint and OpenAI-compatible completion endpoints accept a `watermark` object:

| Field | Type | Behavior |
| --- | --- | --- |
| `enabled` | boolean | `true` uses the server key; `false` opts out unless enforce-all is active. |
| `key` | hex string | Replaces key A and implicitly enables watermarking. |
| `context_window` | integer | Selects a window from 1 through the server maximum. |

| Server mode | Omitted | `{"enabled": false}` | `{"enabled": true}` | `{"key": "HEX"}` |
| --- | --- | --- | --- | --- |
| Watermarking disabled | Off | Off | HTTP 400 | HTTP 400 |
| Enabled, no server key | Off | Off | HTTP 400 | Request key |
| Enabled with server key | Off | Off | Server key | Request key |
| Default-enabled | Server key | Off | Server key | Request key |
| Enforce-all | Server key | HTTP 400 | Server key | Request key |

A request is rejected when it asks for an unavailable key, combines a key with `enabled=false`, exceeds the server context-window limit, or uses a fail-closed incompatible mode. On `/generate`, place `watermark` at the request top level or in `sampling_params`, not both.

## Using the detector

Within an SGLang environment:

```python
from sglang.srt.sampling.watermarking import WatermarkDetector

detector = WatermarkDetector("0123456789abcdef", context_window=4)
result = detector.detect_tokens(completion_ids, prompt_token_ids=prompt_ids)
print(result.combined.num_contexts, result.combined.z_score, result.combined.p_value)
```

For dual-key output, pass `key_b` and the server's `mixing_probability`. Exact generated token IDs are preferred because decoding and re-tokenizing text need not reproduce the original sequence.

The detector package imports only the Python standard library and `msgspec`; NumPy token arrays are accepted but NumPy is optional. It can be copied into another application without SGLang or PyTorch:

```bash
cp -R python/sglang/srt/sampling/watermarking detector_app/watermarking
python -m pip install msgspec
```

```python
from watermarking.detector import WatermarkDetector

detector = WatermarkDetector("0123456789abcdef", context_window=4)
result = detector.detect_tokens(completion_ids, prompt_token_ids=prompt_ids)
```

## Limitations

- Greedy generation carries no watermark signal. Short and low-entropy text provides less statistical power; the entropy gate trades signal for ordinary sampling at low-entropy positions.
- Paraphrasing and translation erase the signal. Edits reduce the number of usable contexts.
- The 32-bit Murmur-derived hash is not a cryptographic PRF. A key holder can detect and forge the watermark.
- Detection requires the generation key, context window, tokenizer, and preferably the exact prompt and completion token IDs.
- Generation stops forcing after its context-history budget. Rare 32-bit context-hash collisions and conservative cross-sibling suppression in wide speculative trees can only reduce signal.
- Unlimited top-k sorts the full vocabulary. Its throughput cost and speculative overhead depend on model, hardware, and workload.
- Unsupported combinations fail closed. The operator guide maintains the compatibility matrix.

## Attribution

The fused Triton selector is adapted from Meta's [TextSeal](https://github.com/facebookresearch/textseal) selector implementation under Apache-2.0 and reimplemented on the hash contract above. Detection follows this contract rather than TextSeal's detector tooling.
