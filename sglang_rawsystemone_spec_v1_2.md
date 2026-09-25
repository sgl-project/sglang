# SGLang rawsystemone: native prefix–suffix scoring

**Implementation specification for Codex · Version 1.2 · September 24, 2026**

**Status:** Implemented on the feature branch with CPU contract tests; GPU parity and performance validation remain pending.

**Revision 1.2:** Require parallel suffix evaluation. Small option sets use one native batch; large option sets use separate, bounded, concurrently submitted batches. This replaces the previous candidate-anchor-first execution plan.

**API amendment · September 25, 2026:** Fix the prefix token boundary, score each suffix conditionally on that prefix, divide by the suffix token count, then apply softmax across the original options. This supersedes the original full-sequence mean and whole-text tokenization rules.

## 1. Implementation assignment

Add the `rawsystemone` completion-scoring endpoint to the **existing SGLang HTTP server**, using its already-loaded causal language model and inference scheduler. Do not introduce a second server, an HTTP proxy back into SGLang, a second copy of the model, or any model training.

The primitive accepts a string `prefix` and a nonempty list of string `suffixes`. Tokenize the prefix once using the native raw-text policy and each suffix separately without added special tokens. Append the token IDs, preserving the same prefix for every option:

```text
P = native_tokenize(prefix)
S[i] = tokenize(suffixes[i], add_special_tokens=False)
L[i] = log P_model(S[i] | P)
     = log P_model(P + S[i]) - log P_model(P)
x[i] = L[i] / len(S[i])
score[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)) for j)
```

**Only suffix tokens contribute to the sum and denominator.** The prefix supplies conditioning context and shared KV state. Compute `L[i]` directly from suffix-token log-probabilities to avoid cancellation error from subtracting large full-sequence sums. `score` is a relative weight over this option list, not calibrated correctness confidence.

Token scores are **log-probabilities after vocabulary-wide log-softmax**, not raw model logits. The option-level softmax uses the conditional mean log-probabilities `x[i]` as its inputs.

**Required execution:** prepare and score the shared token prefix once, then make independent suffix branches eligible for batched/parallel evaluation through the existing scheduler. When the option count or token volume is too large for one submission batch, split it into separate batches and submit multiple batches concurrently with bounded concurrency. Do not await each option, or each batch, in a serial loop. Actual GPU execution remains under the native scheduler; no extra process, model copy, or custom GPU batching engine is required. Section 6 defines the execution contract.

A separate, optional LLM-based compiler translates higher-level dialogue-understanding questions into this representation. It is not called by the scoring endpoint. Section 12 defines that boundary.

### Non-goals

No A/B/C-label prompting, joint option-list prompting, answer generation, SFT, reinforcement learning, decision head, automatic calibration, or Jev protocol compatibility in the core patch. Do not change the loaded checkpoint or existing `/v1/score` behavior. A score is not advertised as a calibrated probability that an interpretation is correct.

## 2. Repository integration and verified starting points

Before editing, record the local SGLang commit, inspect its actual architecture, and adapt the paths below if the checkout differs. These are upstream integration landmarks reviewed on September 24, 2026, not a guarantee about the user's installed version.

| Upstream location | Relevant starting point |
| --- | --- |
| `python/sglang/srt/entrypoints/http_server.py` | Existing FastAPI routes, JSON validation, and serving-handler initialization. [S1] |
| `python/sglang/srt/entrypoints/openai/serving_score.py` | Existing scoring handler; currently delegates to `tokenizer_manager.score_request`. Do not repurpose its scoring semantics. [S2] |
| `python/sglang/srt/managers/io_struct.py` | `GenerateReqInput`, including batched `input_ids`, `return_logprob`, and `logprob_start_len`. [S3] |
| `python/sglang/srt/managers/tokenizer_manager.py` | `generate_request`, native text tokenization, request cancellation, and result metadata. [S4] |
| `python/sglang/srt/managers/schedule_batch.py` | Prefix-cache matching and the predictor/target shift for input-token log-probabilities. [S5] |
| `python/sglang/lang/backend/runtime_endpoint.py` | Existing choice-scoring precedent: cached prefix, batched supplied completions, and zero requested new tokens. This is not the full-prefix scoring contract specified here. [S6] |

**Proposed implementation layout:** introduce a small `RawSystemOneService` and request/response schema module under the local entrypoint layout; register one route and initialize one handler with the existing tokenizer manager. Keep token-position mapping and score aggregation separately unit-testable. Avoid broad refactors or copied scheduler code.

Use `GenerateReqInput` through the in-process manager rather than calling `http://localhost/...`. Do not create a new `Engine` inside a request handler. A separate engine API or SDK method is optional follow-up work, not a prerequisite.

## 3. Proposed HTTP contract

### Naming

- Feature and API name: `rawsystemone`.
- Native route: `POST /v1/rawsystemone`.
- Proposed handler/service: `RawSystemOneService`; request/response schemas: `RawSystemOneRequest` and `RawSystemOneResponse`.
- Response object discriminator: `rawsystemone`.

“Raw” identifies the low-level `prefix` + `suffixes` likelihood-scoring primitive. The optional LLM-to-completion compiler remains outside this endpoint; the name does not imply Jev protocol compatibility. Version 1.1 established this naming. Version 1.2 requires parallel/batched suffix evaluation; the September 25 amendment defines conditional suffix normalization with a fixed token boundary.

### Endpoint

```http
POST /v1/rawsystemone
Content-Type: application/json
```

This is a **new SGLang-native API proposal**, not an OpenAI-standard or existing Jev endpoint. It must coexist with `/v1/score` unchanged.

### Request

```json
{
  "prefix": "Customer: Please move my appointment to Friday.\n\nThe customer's requested appointment operation is",
  "suffixes": [
    " booking.",
    " cancellation.",
    " rescheduling."
  ],
  "return_token_logprobs": false
}
```

Only `prefix` and `suffixes` are required. `return_token_logprobs` defaults to `false`.

Contract:

- `prefix` is a strict string. `suffixes` is a nonempty list of strict strings. Do not coerce numbers, objects, or nulls into strings.
- Preserve text exactly as supplied: no inserted whitespace, trimming, Unicode normalization by the handler, separators, role markers, chat template, answer marker, or added option list. Tokenize the prefix once and append separately encoded suffix IDs; do not re-tokenize the concatenated text.
- Every suffix must encode to at least one token without added special tokens. Empty or zero-token options return `no_option_tokens`. The native prefix must contain at least one token to predict the first suffix token; an empty prefix is allowed only when native tokenization supplies a token such as BOS. Otherwise return `no_prefix_tokens`.
- Duplicate suffixes are permitted. Return one result per original index; internal deduplication is allowed.
- A one-option request is valid and must not require a separate prefix warm-up.
- Reject unknown fields. In particular, do not silently accept `temperature`, `messages`, `context`, `label_token_ids`, arbitrary sampling parameters, or a different normalization rule.
- Version 1 uses the already-loaded default model. Do not add model downloads, runtime model selection, or per-request adapter selection.

Use the existing server's request-body and inference-admission limits. Add a documented configurable maximum option count, with **128 as the proposed default**, and a bounded cumulative candidate-token budget. Validate the entire request before starting GPU work. Never silently truncate a candidate to fit the context window.

**The maximum request option count is not the internal batch size.** Valid requests larger than one internal batch are automatically partitioned and executed as separate concurrent batches (Section 6.5); the caller still makes one HTTP request and receives one ordered response. Concurrency and batch limits are server configuration, not new required request fields.

### Response

The following values are **illustrative schema examples, not measured model outputs**:

```json
{
  "id": "rawsystemone-example",
  "object": "rawsystemone",
  "model": "served-model-name",
  "scoring": "softmax_mean_suffix_logprob",
  "tokenization": "prefix_suffix_tokens_v1",
  "data": [
    {"index": 0, "score": 0.1863237232258476, "logprob_sum": -4.0, "option_token_count": 2},
    {"index": 1, "score": 0.3071958857184984, "logprob_sum": -3.0, "option_token_count": 2},
    {"index": 2, "score": 0.506480391055654, "logprob_sum": -2.0, "option_token_count": 2}
  ],
  "best_index": 2,
  "usage": {
    "input_tokens": 63,
    "scored_tokens": 60,
    "generated_tokens": 0
  }
}
```

`data` retains input order. Higher `score` is better. `best_index` is the first original index attaining the maximum returned score; exact ties are not resolved through another model call. Do not round scores before ranking.

`option_token_count` is the sole per-candidate token count and the denominator of the conditional mean `x[i]`. `logprob_sum` is `log P_model(S[i] | P)`. `score` is the softmax weight, in `[0, 1]`; weights sum to one within floating-point precision. Expand deduplicated inputs before normalizing: each original option, including duplicates, receives a weight. One option receives `1.0`.

Usage fields are **logical candidate totals**, including duplicate candidates and repeated prefixes. `input_tokens` counts complete `P + S[i]` sequences; `scored_tokens` counts all native scoreable positions, including prefix positions, and equals `input_tokens - len(suffixes)`. It is not the sum of `option_token_count`. These counters do not measure actual GPU computation. Report physical cache/compute statistics separately through diagnostics or metrics.

When token details are requested, each result additionally contains:

```json
{
  "token_logprobs": [
    {"position": 0, "token_id": 123, "logprob": null},
    {"position": 1, "token_id": 456, "logprob": -0.75}
  ]
}
```

Positions are absolute positions in that candidate's complete token sequence. Return all positions, including the unscored initial position. Token text is unnecessary. Do not echo the prefix or suffix by default.

The option-level softmax is required. It produces relative weights over the supplied list, not calibrated confidence or unconditional probabilities for the option strings. Adding an option changes the normalized weights, while each existing option's conditional log-probability sum and token count remain unchanged.

## 4. Exact mathematical and tokenization semantics

First freeze the prompt using the native raw-text tokenizer policy. For candidate `i`, append the suffix IDs without adding another BOS/EOS or applying a chat template:

```text
P = native_tokenize(prefix)
S_i = tokenize(suffixes[i], add_special_tokens=False)
tokens_i = P + S_i
K = len(P)
N_i = len(S_i)
n_i = K + N_i
```

The fixed-boundary token sequence may differ from whole-text tokenization:

```text
native_tokenize(prefix + suffixes[i]) != P + S_i
```

That difference is intentional: appending an option cannot change the last prefix token or its KV state. Preserve the native prefix's configured special tokens, and disable automatic special tokens only for the suffix encoding. Literal special-token strings in supplied text still follow the tokenizer's policy. Document the actual tokenizer/checkpoint configuration used in tests. The existing manager's raw-text path supplies the prefix IDs. [S4]

For a standard causal model, position `t` is predicted from positions before `t`:

```text
ell_i[t] = log_softmax(model_logits_at_position(t - 1))[tokens_i[t]]
           for t = 1, ..., n_i - 1

ell_i[0] = null

L_i = sum(ell_i[K:])
    = log P_model(P + S_i) - log P_model(P)
x_i = L_i / N_i
score_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

The prefix must contain at least one token, so every suffix token has a predictor. Position zero is the only unscored native position and belongs to the prefix. Prefix scores may be retained in optional complete token records, but they contribute to neither `L_i` nor `N_i`. Compute the suffix sum directly rather than subtracting two large totals.

Rules:

- Require `K > 0` and `N_i > 0`.
- `0.0` is a valid log-probability and counts in the denominator. Never filter using truthiness such as `if value`.
- Apart from the expected initial null, missing token scores are errors, not permission to reduce the denominator.
- Reject NaN or non-finite results with a typed inference error in version 1; do not clamp or silently omit them.
- Use vocabulary-wide log-softmax at temperature 1, without top-k/top-p renormalization, penalties, logit bias, grammar constraints, or custom logit processors.
- Accumulate returned token scores with `math.fsum` or equivalent stable float64 host accumulation. Do not change the inference model's precision merely to perform aggregation.
- Count each suffix token once. Overlapping boundary scores from internal calls are bookkeeping, not extra evidence.
- Use the maximum-shifted softmax and stable summation. Very small relative weights may underflow to zero; the maximum option always contributes `exp(0) = 1`, preventing a zero denominator.

Subtracting the prefix contribution must happen **before** length normalization. Dividing full-sequence sums by different option lengths would leave a different prefix term in each score and would not compute conditional suffix means.

## 5. Correctness-first reference implementation

First implement a private reference path used by tests:

1. Tokenize the prefix once with the native policy, encode each suffix without added special tokens, and append the token IDs.
2. Submit each complete token sequence with input-token log-probabilities requested from the beginning and zero requested new tokens.
3. Convert native results into absolute token-position records.
4. Check position coverage and token IDs, then aggregate exactly as in Section 4.

The reference path must not reuse the new service's shared-prefix score bookkeeping. A sequential reference is permitted only as a test oracle/benchmark control. A production reference fallback must still use native batching and bounded concurrent submission of separate batches, subject to admission limits. Test cache-cold and cache-disabled configurations where supported.

Suggested internal request settings, adapted to the checked-out version:

```python
GenerateReqInput(
    input_ids=complete_token_ids,
    sampling_params={"max_new_tokens": 0, "temperature": 1.0},
    return_logprob=True,
    logprob_start_len=0,
    top_logprobs_num=0,
    return_text_in_logprobs=False,
    stream=False,
)
```

Explicitly neutralize any inherited generation defaults that could alter the likelihood distribution. Verify behavior against an independent teacher-forced causal-model calculation on a small unquantized checkpoint. Do not assume setting `temperature=0` is appropriate for probability extraction.

The upstream native API documents the log-probability controls; the current frontend demonstrates supplied-text scoring with `max_new_tokens=0`. [S6][S7] This specification requires consuming the supplied input-token scores, not generated answer-token scores.

## 6. Optimized path: shared prefix, parallel suffixes

### 6.1 Plan sharing and batching in token space

Tokenize the prefix once and append separately encoded suffix IDs, then deduplicate identical complete token-ID sequences. Let `K = len(P)` be the fixed prefix boundary. This boundary does not depend on the set of candidate options.

Keep `K` fixed even when different suffixes start with identical tokens: those tokens still belong to each suffix's conditional score. A suffix must never see a different candidate in its causal input. Use a deterministic internal ordering, such as sequence length followed by token-ID lexicographic order, for reproducible planning. Preserve a mapping back to every original index.

Plan the common span across the **entire request before partitioning it into submission batches**. All batches reuse the same request-local shared likelihood records. Do not independently prefill or score the same shared prefix for each submission batch.

### 6.2 Score the common prefix once, then release every branch

For multiple distinct candidates and a useful shared span (`K > 1`), submit `P` for input-logprob scoring with zero requested new tokens. This is a prefix-only operation, not the scoring of an arbitrarily selected candidate. Pass the existing token IDs directly; do not decode and re-tokenize them or insert special tokens again.

Retain the prefix token-position records for complete token diagnostics. Their sum is the term excluded from the candidate score:

```text
L_shared = sum(prefix_logprobs[t] for 1 <= t < K)
```

After that shared operation completes and its results have been validated, all nonempty independent branch tails become eligible together for the bounded batch dispatcher in Section 6.5. **No branch depends on another candidate's score or completion.** In particular, do not wait for one full candidate's suffix before admitting the other suffixes.

For every distinct candidate with a tail, request only the log-probabilities needed for these target positions, while supplying its complete token-ID sequence as required by the native manager:

```text
first_new_target = K
L_tail_i = sum(logprobs_i[t] for K <= t < n_i)
N_tail_i = n_i - K

L_i = L_tail_i              # prefix contribution excluded
N_i = N_tail_i              # equals len(S_i)
x_i = L_i / N_i
score_i = softmax(x)[i]     # after expanding duplicates to original indices
```

Reuse the warmed KV cache where supported, and retain the shared likelihood values separately from KV state. Section 6.3 defines the predictor-boundary handling needed for the first tail token.

Special cases:

- **One unique complete sequence:** score it once without a separate prefix-only call, then expand the result to duplicate original indices.
- **A zero-token option:** reject before prefill; every accepted candidate has a suffix tail.
- **`K <= 1`, caching disabled, or unsupported span reuse:** skip unnecessary prefix warm-up and use the full-sequence reference fallback with native batching and concurrent batch submission. Do not silently switch to a serial option loop.

The prefix-only phase is an explicit dependency needed for score/cache reuse. Parallelism applies to the independent branch phase. This revision replaces the earlier full-candidate-anchor-first plan.

### 6.3 Respect the predictor/target boundary

A target at absolute position `K` needs the model output at `K - 1`. The current upstream scheduler shifts input log-probability targets by one, and constrains cache matching according to `logprob_start_len`. [S5]

For this source layout, `max(K - 1, 0)` is the expected internal start for obtaining the first branch target. **Verify this against the local revision with token-position tests before relying on it.** Native response conventions may include an initial null or boundary overlap.

Implement a small adapter that maps native rows into `(absolute_position, token_id, logprob)` and verifies coverage. Never align by token text, search for the first matching token ID, or discard a fixed number of rows without a tested offset contract. Repeated tokens must work correctly.

It is acceptable to recompute the final shared predictor token to obtain the first suffix score. It is not acceptable to omit the first suffix token or double-count the last shared token.

### 6.4 KV cache and likelihood values are different things

The existing KV cache does not, by itself, supply previously requested scalar token log-probabilities. Retain the shared likelihood values or their aggregate in this request's service state. Do not repeatedly request scores from position zero for every candidate and call that an optimized implementation.

Start with request-local score reuse. A persistent global prefix-score cache is **out of scope** because it requires additional invalidation, versioning, privacy isolation, and memory limits.

### 6.5 Large option lists: separate batches submitted concurrently

**Small request:** submit all distinct branch candidates together through the manager's supported native batch path.

**Large request:** partition the candidates into separate submission batches, then keep several batches in flight concurrently. Partition by both candidate count and token volume, not option count alone. Long individual suffixes must not make a nominally small batch unbounded.

```text
                       shared prefix scored once
                                  |
                  bounded concurrent batch dispatcher
                     /            |            \
                  batch 0      batch 1       batch 2 ...
                 suffixes      suffixes      suffixes
                     \            |            /
                   collect by stable candidate/index ID
                                  |
                       one ordered API response
```

Required behavior:

1. **Use the existing manager and scheduler.** Each submission batch is a group of independent full token-ID inputs with the appropriate logprob spans. Do not concatenate candidates into a single causal sequence. Use native batching, or bounded concurrent child requests if the local manager lacks a safe batched interface.
2. **Bound each batch.** Add documented internal limits for candidate count and token volume. A proposed initial `max_candidates_per_batch` is **32**; it is a configurable starting point, not a measured optimum. Token budgets must fit the deployment's existing context/admission limits. Account for full logical inputs and potential cache-miss work, not only an optimistic cached-tail estimate.
3. **Bound concurrency.** A proposed initial `max_inflight_batches_per_request` is **2**. Also enforce a server-wide bound/fair admission policy across parent requests, including a bound on outstanding candidate/token work. These settings must not let many concurrent HTTP requests each flood the scheduler. They are proposed internal configuration names, not existing SGLang flags.
4. **Keep the dispatcher work-conserving.** As soon as any in-flight batch completes and admission permits, submit the next pending batch. Do not wait for every batch in a wave to finish. Do not implement `for batch in batches: await score_batch(batch)`, and do not use an unbounded gather over all options/batches.
5. **Preserve independent likelihoods.** Batch membership, execution order, and other options must not change an option's conditional sum or token count. Store results by stable candidate ID, expand duplicates, restore original input order, then apply softmax across the original option list. Adding options changes the normalized weights. Rank only after every required result is valid.
6. **Keep one shared prefix state per parent request.** All batches reuse the same prefix-logprob records and, when available, the same native cache location. Splitting the list must not create a new prefix-scoring phase for each batch.
7. **Handle large individual candidates.** A candidate over the preferred submission token target may run as a singleton batch only if it is within the hard context, token, and admission limits. Leave chunked prefill to the existing scheduler. Never split a suffix into independent texts or truncate it to make a batch fit.

For example, **96 distinct suffixes** with a 32-candidate batch limit produce three batches. With two batch slots, submit batches 0 and 1 concurrently; submit batch 2 when either slot becomes available. This is an illustrative scheduling example, not a latency or GPU-utilization claim.

Use a bounded queue/worker pool or the local runtime's equivalent. Within a batch, await one native batched operation or concurrently consume bounded child generators; merely packaging a serial per-option loop in a batch helper does not satisfy this requirement. All admission leases and task/generator resources must be released on both success and failure.

### 6.6 Parallel submission is not a promise of simultaneous GPU kernels

This contract requires scheduler-visible batching/concurrency, not one Python thread, OS process, CUDA stream, or model replica per option. Separate submission batches may be merged, split into microbatches, interleaved with other traffic, or run sequentially by the native scheduler under memory/compute pressure. Do not override native scheduling to force physical simultaneity or claim that `async` alone proves GPU overlap.

A configured concurrency/admission limit of one may serialize execution on a constrained deployment; record that limitation rather than claiming measured parallel execution. On a test configuration with at least two available batch slots, prove that two independent batches can be submitted before either finishes.

### 6.7 Cache misses, routing, and cancellation

Cache reuse is an optimization, not a correctness dependency. Eviction, page alignment, or scheduling may force additional computation. Scores and scoring spans must remain unchanged. Fall back to correct full-sequence batched scoring when span reuse is unsupported, and report the fallback reason. The parallel batch dispatcher still applies.

For data-parallel deployments, prefer supported worker-affinity/routing mechanisms to keep related batches near the warmed cache. Do not assume KV cache is shared across independent replicas. If existing routing sends branches to another worker, correctness must still hold and extra prefill/cache misses must be measured. Do not create additional model replicas for this feature. Tensor-parallel and chunked-prefill modes require explicit parity tests.

On disconnect, timeout, cancellation, or any child/batch failure, stop dispatching queued batches, cancel/abort all in-flight children, close generators, and release every batch/admission slot. Do not return a partial list as a successful response or leave other batches running after the parent has failed.

## 7. Optional single-token fast path

After the cached multi-token path passes all tests, add a fast path when every candidate is the same token prefix plus exactly one token.

The desired computation is one shared-prefix evaluation, followed by gathering the candidate token log-probabilities from the same next-token distribution. These are the conditional suffix sums; do not add `L_shared`. Apply the required option-level softmax after expanding duplicate options.

Each selected token must first retain its **full-vocabulary** log-probability in `logprob_sum`. The final option-level softmax normalizes the conditional means; do not overwrite the underlying token log-probabilities with that option normalization.

Use this optimization only when the checked-out backend exposes the required raw selected-token scores without an answer-generation loop. Otherwise retain the general cached path. Tokenizer-added trailing tokens, unequal residual lengths, or a boundary mismatch make the one-token optimization inapplicable.

This fast path is optional for the first patch. Correctness and verified shared-prefix reuse are required.

## 8. Server lifecycle, compatibility, and failure handling

Use the same HTTP process, listening port, authentication middleware, scheduler admission, tracing, and shutdown lifecycle as normal inference.

The first implementation supports text-only causal likelihood scoring on compatible loaded models. Reject unsupported embedding/classification/diffusion modes explicitly; do not substitute a different kind of score. More specialized model architectures or disaggregated deployments must either pass the relevant tests, use a correct reference fallback, or return a documented unsupported-mode error.

All candidates in one response must use one model-weight snapshot and tokenizer configuration. Reuse existing weight-update synchronization safely, or detect a successful update using a reliable monotonic runtime epoch and fail the whole composite request. Model-path equality alone is insufficient. Do not add a nested read-lock scheme that can deadlock when a writer is waiting.

Give internal requests unique child IDs. On client disconnect, timeout, cancellation, or branch failure, cancel/abort all outstanding children and close their async generators through existing mechanisms. Do not leave GPU work running or return a partial score list as a successful response.

Preserve request privacy and cache isolation settings through every child request, including no-log controls and cache salts where the local server uses them. Do not log prompts, suffixes, or token traces by default.

Use the server's existing error envelope and status conventions, with stable machine-readable error codes. Cover invalid field types, zero scoreable tokens, excessive options, cumulative token budget, context overflow, unsupported model/mode, unavailable tokenizer, incomplete/non-finite scores, overload, and model changes during evaluation.

No unrelated endpoint or global validation behavior may change.

## 9. Observability and performance acceptance

Record option count, distinct candidate count, shared-token span, logical input tokens, actual cache-hit/processed-token counters when available, endpoint latency, scoring mode, and reference-fallback reason. Also record submission batch count, per-batch candidate/token counts, configured and observed in-flight batch counts, queue wait time, shared-prefix time, and branch-phase time. Use native execution traces where available to distinguish concurrent submission from actual GPU batching. Separate compiler latency from scorer latency.

For an effective shared span, the intended work pattern is:

```text
one shared-prefix scoring/prefill phase
+ parallel branch-tail scoring (one batch or bounded concurrent batches)
+ small boundary/alignment overhead
```

Do not claim latency scales only with suffix length: each suffix still attends to its preceding context, and cache availability, model architecture, scheduling, and vocabulary projection costs matter.

Benchmark the reference and optimized implementations on the **same checkpoint, precision, tokenizer, hardware, and server settings**. Suggested matrix:

| Dimension | Cases |
| --- | --- |
| Prefix length | Approximately 128, 1,024, and 4,096 tokens, within model limits |
| Option count | 2, 8, 32, 96, and the configured request limit |
| Suffix length | Approximately 1, 4, and 16 tokens, plus mixed short/long suffixes |
| Batch settings | One native batch, split batches with concurrency 1, 2, and 4 where admission permits |
| Cache conditions | Cold, warm, and disabled where supported |
| Traffic | Single request and concurrent mixed inference |

Publish median and p95 latency, throughput, physical cache/compute measurements, and score differences. Record warm-up and repetition counts. Compare sequential full-sequence scoring, one native batch, and split concurrent batches on the same workload. Performance acceptance requires **demonstrated prefix reuse and lower redundant prefill work on long-prefix/multiple-option cases**, plus verified non-serial batch dispatch when capacity is available. Do not invent a universal speedup target or assume that more concurrent batches always improve latency.

## 10. Required test suite

### A. Pure unit tests

1. Fixed prefix tokenization, exact suffix whitespace, nonempty token prefix/options, and rejection of unknown fields.
2. Conditional suffix aggregation excludes prefix scores; unequal lengths use each suffix's token count, followed by a stable option-level softmax.
3. A log-probability of `0.0` counts; only the expected initial null is excluded. Other missing or non-finite values fail.
4. Token-space deduplication with one option, all-identical options, short prefixes, and shared option text that must remain inside the suffix score.
5. Return-order preservation and exact-tie policy.
6. Mocked native results with boundary overlap, repeated token IDs, and chunked returns map to correct absolute positions.
7. Cancellation and partial branch failure release every child request, batch slot, and admission lease, and stop queued batches.
8. Batch partitioning honors count/token bounds, covers every distinct candidate exactly once, and preserves duplicate-to-original mappings.
9. A gated mock dispatcher proves concurrency: block batch A and verify that batch B starts before A is released when two slots are available. Test observed in-flight counts never exceed per-request/global limits.
10. Release a short batch while a long batch remains blocked and verify that the next pending batch starts immediately; reject wave-barrier/serial-loop behavior.
11. Verify one shared-prefix scoring call for a split-batch request, no branch dependency on another full candidate, and no redundant warm-up for one unique candidate.
12. Exercise cancellation while queued, waiting for admission, consuming a batch result, and after one batch fails. No new batch starts after cancellation.

### B. GPU/integration parity tests

1. **Reference equality:** each optimized candidate matches independent full-sequence scoring on the same server configuration.
2. **Independent oracle:** a small supported unquantized checkpoint agrees with teacher-forced causal-model log-softmax calculations using identical token IDs and special-token policy.
3. **Permutation invariance:** shuffling options preserves each candidate's score within numerical tolerance after mapping back. Exact ties may change `best_index` by the documented input-order rule.
4. **Candidate-set independence:** conditional `logprob_sum` and `option_token_count` remain unchanged beside duplicates or distractors, within established numerical tolerance. Final softmax weights change with the option list; a single option has weight `1.0` and all-identical options share weight equally.
5. **Tokenizer boundaries:** leading spaces, trailing spaces, punctuation, split words, newline joins, Unicode, and byte-level/multibyte cases. Verify a case where whole-text tokenization differs and confirm that inference uses the fixed prefix IDs plus separately encoded suffix IDs.
6. **First-token coverage:** BOS and no-BOS behavior, very short sequences, and the first differing suffix token are handled correctly.
7. **Unequal lengths:** confirm that the prefix contribution is excluded before division by suffix length and that all final weights sum to one within floating-point precision.
8. **Caching:** cold, warm, disabled, and eviction scenarios return equivalent scores. Confirm that cache reuse really occurs in a supported optimized case.
9. **Batching and concurrency:** compare sequential oracle, single native batch, and separate batches at concurrency 1, 2, and 4 where supported. Test large lists, mixed lengths, singleton batches, duplicate options, and options moved between batch boundaries. All candidate scores must match within the established tolerance.
10. **Execution modes:** chunked prefill and supported tensor-parallel configurations. Test other modes before claiming support.
11. **Lifecycle:** HTTP disconnect, timeout, admission rejection, invalid context length, and simulated weight updates between shared-prefix scoring and branches or between separate batches.
12. **Regression:** `/generate`, chat/completions, `/v1/score`, authentication, and startup remain functional. Generated decision text is never used for scoring.
13. **Scheduler-visible concurrency:** use manager/scheduler traces to show multiple branches or submission batches in flight when resources permit; report actual device batch behavior separately. Test mixed traffic for bounded admission and cleanup without overloading normal inference.

Pure aggregation tests must be exact. Numerical integration tolerances must be documented per dtype/backend and established against independent full scoring. Start with tight tolerances on the small unquantized oracle. Do not silently loosen tolerances to hide an indexing bug; report maximum per-token and per-sequence differences and explain justified hardware-dependent differences. Quantized cross-backend comparisons are not substitutes for same-backend parity.

## 11. Suggested implementation sequence and deliverables

**First:** inspect the local commit and native input-logprob behavior; write token-position and full-sequence reference tests.

**Second:** add schemas, the native route, a reusable scoring service, and the correct reference path. Verify startup and legacy endpoints.

**Third:** implement token-space sharing, one shared-prefix scoring/prefill phase, request-local prefix-score retention, native branch batching, and bounded concurrent dispatch of separate batches for large option lists. Add fail-fast cancellation and admission cleanup. Verify parity and concurrency before optimizing further.

**Fourth:** document the API and publish reproducible benchmarks. Add the optional single-token fast path only after the general implementation is correct.

Deliver:

- A focused patch to the existing SGLang server, without new model weights or a second serving process.
- Unit and integration tests plus exact commands to run them on the local checkout.
- API documentation and a runnable client example.
- Benchmark script/results identifying commit, model revision, tokenizer, precision, hardware, and supported modes.
- An implementation report listing completed work, measured results, and any explicitly unsupported configuration. Do not present unrun tests as passing.

A request example for the proposed endpoint, **after the patch is implemented**:

```bash
curl http://127.0.0.1:30000/v1/rawsystemone \
  -H 'Content-Type: application/json' \
  -d '{
    "prefix": "Customer: Please cancel my appointment.\n\nThe requested operation is",
    "suffixes": [" booking.", " cancellation.", " rescheduling."]
  }'
```

## 12. Companion specification: LLM-to-completion compiler

This is the higher-level dialogue-understanding layer discussed alongside the native primitive. Keep it as an optional client/example module using an already-configured LLM; it must not become an inference-server dependency or a hidden generation call inside `/v1/rawsystemone`.

### Compiler input

```json
{
  "question": "What appointment operation is the customer requesting?",
  "options": [
    {"id": "book", "meaning": "Create a new appointment"},
    {"id": "cancel", "meaning": "Cancel an existing appointment"},
    {"id": "reschedule", "meaning": "Change the time of an existing appointment"}
  ]
}
```

### Compiled artifact

```json
{
  "version": 1,
  "statement_prefix": "The customer's requested appointment operation is",
  "suffixes": [" booking.", " cancellation.", " rescheduling."],
  "option_ids": ["book", "cancel", "reschedule"]
}
```

The compiler expresses all interpretations as parallel candidate completions. It must not choose the answer. Preserve option meaning, negation, conditions, and one-to-one ID correspondence. Avoid A/B/C labels, persuasive candidate explanations, and listing competing options inside the scored prefix.

A deterministic renderer combines the current context with the compiled artifact:

```python
prefix = context + "\n\n" + compiled.statement_prefix
suffixes = compiled.suffixes
```

The renderer preserves the original context verbatim. The resulting context text is part of `prefix` and therefore **is included in the score**, exactly as the native API requires.

Validate the compiler's structured output: schema, types, exact option count, exact ID correspondence, order, and nonempty usable completions. Structural validation does not prove semantic fidelity; use review and held-out/paraphrase tests for that.

For recurring DU decisions, compile once at configuration time, version the artifact, and reuse it at runtime. Record the source decision schema, compiler model/prompt version, and template hash. A changed question or option set requires an explicit new compiled version; do not silently regenerate templates during normal execution.

For genuinely dynamic questions, compile on demand outside the scorer and measure that cost separately. The compiler model and scoring model may differ; compiling a template does not update the scoring model's weights.

Code maps the returned result index back to the original option ID. The application's state machine remains responsible for legal transitions, clarification, and authorization of side effects. A returned likelihood score does not itself authorize an appointment change. Open-ended values require candidate generation or extraction before this finite comparison can select them.

**Optional companion deliverable:** a small `compile -> validate -> render -> score -> map` example with a pluggable existing LLM client. Do not hard-code a provider, add mandatory API credentials to the server, or delay the native scoring patch to build a Jev compatibility layer.

## 13. Definition of done

The feature is complete when a caller can send only a prefix and suffix list to the existing SGLang server and obtain auditable conditional suffix log-probabilities and softmax weights over their token-length-normalized values, with fixed prefix tokenization, proven reference parity, and shared-prefix reuse on supported configurations. Small option sets use native batching; large option sets are automatically split into bounded batches submitted concurrently under available admission capacity. The implementation must prove it does not serialize independent options/batches in the handler, and must restore all results to input order.

The implementation must remain correct with caching unavailable, must preserve candidate independence and original result order, and must make its token-count convention explicit. No second server, answer-generation protocol, or model training is needed to use the primitive.

## Sources and verification notes

Revision 1.2 updates the proposed execution requirements; it is not a claim that concurrent-batch support for this endpoint has been implemented or benchmarked. The source landmarks below are retained from the earlier specification and must be checked against the implementation checkout. The sources below establish upstream interfaces and implementation landmarks, not the existence or performance of the proposed endpoint. The source URLs track `main`; the implementer must pin and report the actual checked-out commit. No SGLang GPU benchmark or server patch was executed while preparing this specification.

[S1] SGLang upstream HTTP server, route and serving-handler initialization.
`https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/entrypoints/http_server.py`

[S2] SGLang upstream existing scoring handler.
`https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/entrypoints/openai/serving_score.py`

[S3] SGLang upstream internal request structures, especially `GenerateReqInput`.
`https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/managers/io_struct.py`

[S4] SGLang upstream tokenizer manager, including native tokenization and request handling.
`https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/managers/tokenizer_manager.py`

[S5] SGLang upstream scheduler request/batch implementation, including `_compute_max_prefix_len` and shifted input-logprob target construction.
`https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/managers/schedule_batch.py`

[S6] SGLang upstream frontend runtime endpoint, especially `RuntimeEndpoint.select`.
`https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/lang/backend/runtime_endpoint.py`

[S7] Official SGLang sampling/request parameter documentation.
`https://docs.sglang.io/docs/basic_usage/sampling_params`
