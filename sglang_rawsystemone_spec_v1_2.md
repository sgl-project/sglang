# SGLang rawsystemone: native prefix–suffix scoring

**Implementation specification for Codex · Version 1.2 · September 24, 2026**

**Status:** Proposed change, not an implemented or benchmarked server feature.

**Revision 1.2:** Require parallel suffix evaluation. Small option sets use one native batch; large option sets use separate, bounded, concurrently submitted batches. Shared-prefix scores and the full-sequence scoring formula are unchanged. This replaces the previous candidate-anchor-first execution plan.

## 1. Implementation assignment

Add the `rawsystemone` completion-scoring endpoint to the **existing SGLang HTTP server**, using its already-loaded causal language model and inference scheduler. Do not introduce a second server, an HTTP proxy back into SGLang, a second copy of the model, or any model training.

The primitive accepts a string `prefix` and a nonempty list of string `suffixes`. For every suffix, independently evaluate the complete concatenation and return its mean token log-probability:

```text
score[i] = logprob_sum(tokenize(prefix + suffixes[i]))
           / number_of_scored_tokens(tokenize(prefix + suffixes[i]))
```

**Both the sum and denominator include the entire prefix and the suffix.** This includes conversation/context text placed in `prefix`; the server must not silently treat any part of it as unscored conditioning context.

“Logit” in the original discussion means the actual token’s **log-probability after vocabulary-wide log-softmax**, not a raw logit. Preserve this distinction in code, field names, documentation, and tests.

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

“Raw” identifies the low-level `prefix` + `suffixes` likelihood-scoring primitive. The optional LLM-to-completion compiler remains outside this endpoint; the name does not imply Jev protocol compatibility. Version 1.1 established this naming. Version 1.2 changes execution to mandatory parallel/batched suffix evaluation, not the scoring semantics or compiler boundary.

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
- Concatenate exactly as supplied: no inserted whitespace, trimming, Unicode normalization by the handler, separators, role markers, chat template, answer marker, or added option list.
- Empty prefixes and empty suffixes are permitted when the resulting token sequence has at least one scoreable token. Validate every complete candidate.
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
  "scoring": "mean_logprob_full_sequence",
  "tokenization": "native_text_v1",
  "data": [
    {"index": 0, "score": -1.8, "logprob_sum": -36.0, "scored_token_count": 20},
    {"index": 1, "score": -1.7, "logprob_sum": -34.0, "scored_token_count": 20},
    {"index": 2, "score": -1.5, "logprob_sum": -30.0, "scored_token_count": 20}
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

`scored_token_count` is the sole per-candidate token count and the denominator of `score`. The complete native token count is always `scored_token_count + 1`, so it is not returned separately.

Usage fields are **logical candidate totals**, including duplicate candidates and repeated prefixes. They do not claim to measure actual GPU computation. Report physical cache/compute statistics separately through diagnostics or metrics.

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

No softmax distribution is required. A future caller may compute normalized relative weights, but these must not be named calibrated confidence or change this score definition.

## 4. Exact mathematical and tokenization semantics

For candidate `i`, obtain the complete token sequence using the same raw-text tokenizer behavior as the server's native text inference path:

```text
x_i = native_tokenize(prefix + suffixes[i])
n_i = len(x_i)
```

Do **not** assume:

```text
tokenize(prefix + suffix) == tokenize(prefix) + tokenize(suffix)
```

The native tokenizer may add configured special tokens. Preserve that existing policy consistently; do not manually add another BOS or EOS. No chat-template application is allowed. Document the actual tokenizer/checkpoint configuration used in tests. The existing manager's raw-text path uses the configured tokenizer directly. [S4]

For a standard causal model, position `t` is predicted from positions before `t`:

```text
ell_i[t] = log_softmax(model_logits_at_position(t - 1))[x_i[t]]
           for t = 1, ..., n_i - 1

ell_i[0] = null

L_i = sum(ell_i[1:])
N_i = n_i - 1
score_i = L_i / N_i
```

Thus, “all tokens” means **all scoreable tokens in the native complete sequence**. The initial token has no earlier model position and is excluded from numerator and denominator. When the tokenizer inserts a leading BOS, it normally occupies that initial position, allowing the first text token to be scored. Any tokenizer-added token at a later position is included under this native policy; the handler must not selectively remove its score.

Rules:

- Require `N_i > 0`.
- `0.0` is a valid log-probability and counts in the denominator. Never filter using truthiness such as `if value`.
- Apart from the expected initial null, missing token scores are errors, not permission to reduce the denominator.
- Reject NaN or non-finite results with a typed inference error in version 1; do not clamp or silently omit them.
- Use vocabulary-wide log-softmax at temperature 1, without top-k/top-p renormalization, penalties, logit bias, grammar constraints, or custom logit processors.
- Accumulate returned token scores with `math.fsum` or equivalent stable float64 host accumulation. Do not change the inference model's precision merely to perform aggregation.
- Count each token once. Overlapping boundary scores from internal calls are bookkeeping, not extra evidence.

For unequal candidate lengths, the prefix's contribution does not generally cancel after averaging. Preserve the requested formula anyway; do not “correct” it to suffix-only likelihood, joint log-likelihood without normalization, or a different statistical objective.

## 5. Correctness-first reference implementation

First implement a private reference path used by tests:

1. Tokenize every complete concatenation using the canonical policy.
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

Batch-tokenize the **full concatenated strings**, then deduplicate identical complete token-ID sequences. Let `K` be the longest common token-prefix length of the unique sequences. Compute it across complete sequences, not from the separately tokenized prefix.

This is safe even when the shared token span extends into the textual suffixes. A suffix must never see a different candidate in its causal input. Use a deterministic internal ordering, such as sequence length followed by token-ID lexicographic order, for reproducible planning. Preserve a mapping back to every original index.

Plan the common span across the **entire request before partitioning it into submission batches**. All batches reuse the same request-local shared likelihood records. Do not independently prefill or score the same shared prefix for each submission batch.

### 6.2 Score the common prefix once, then release every branch

For multiple distinct candidates and a useful shared span (`K > 1`), submit the common token-ID sequence `x_0[:K]` for input-logprob scoring with zero requested new tokens. This is a prefix-only operation, not the scoring of an arbitrarily selected candidate. Pass the existing token IDs directly; do not decode and re-tokenize them or insert special tokens again.

Retain the common token-position scores and their aggregate in request-local state:

```text
L_shared = sum(prefix_logprobs[t] for 1 <= t < K)
N_shared = K - 1
```

After that shared operation completes and its results have been validated, all nonempty independent branch tails become eligible together for the bounded batch dispatcher in Section 6.5. **No branch depends on another candidate's score or completion.** In particular, do not wait for one full candidate's suffix before admitting the other suffixes.

For every distinct candidate with a tail, request only the log-probabilities needed for these target positions, while supplying its complete token-ID sequence as required by the native manager:

```text
first_new_target = K
L_tail_i = sum(logprobs_i[t] for K <= t < n_i)
N_tail_i = n_i - K

L_i = L_shared + L_tail_i
N_i = N_shared + N_tail_i    # equals n_i - 1
score_i = L_i / N_i
```

Reuse the warmed KV cache where supported, and retain the shared likelihood values separately from KV state. Section 6.3 defines the predictor-boundary handling needed for the first tail token.

Special cases:

- **One unique complete sequence:** score it once without a separate prefix-only call, then expand the result to duplicate original indices.
- **A candidate ends at the common span:** its complete result comes from the shared records; no branch inference is needed.
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
5. **Preserve independent scores.** Batch membership, execution order, and the number of other candidates must not change the scoring formula. Store results by stable candidate ID, expand duplicates, and restore original input order. Rank only after every required result is valid.
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

The desired computation is one shared-prefix evaluation, followed by gathering the candidate token log-probabilities from the same next-token distribution and combining them with `L_shared`.

Crucially, each selected token must retain its **full-vocabulary** log-probability. Renormalizing only over the candidate token set changes the specified sequence score and is prohibited.

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

1. Exact concatenation, validation, nonempty option list, and rejection of unknown fields.
2. Score aggregation includes prefix and suffix; unequal lengths use each candidate's actual denominator.
3. A log-probability of `0.0` counts; only the expected initial null is excluded. Other missing or non-finite values fail.
4. Token-prefix planning with duplicates, no shared span, one option, all-identical options, and a candidate that is a token prefix of another.
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
4. **Candidate-set independence:** scoring a candidate alone, beside duplicates, or beside additional distractors does not semantically change its score. Allow only established floating-point execution variation.
5. **Tokenizer boundaries:** leading spaces, trailing spaces, punctuation, split words, newline joins, Unicode, and byte-level/multibyte cases. Include a verified case where separately encoding prefix and suffix differs from encoding their concatenation.
6. **First-token coverage:** BOS and no-BOS behavior, very short sequences, and the first differing suffix token are handled correctly.
7. **Unequal lengths:** confirm that including the prefix produces the expected full-sequence average, not suffix-only normalization.
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

The feature is complete when a caller can send only a prefix and suffix list to the existing SGLang server and obtain auditable full-sequence mean log-probabilities, with proven reference parity and shared-prefix reuse on supported configurations. Small option sets use native batching; large option sets are automatically split into bounded batches submitted concurrently under available admission capacity. The implementation must prove it does not serialize independent options/batches in the handler, and must restore all results to input order.

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
