Status: implemented

# Sampling masks

TL;DR: Requests with `return_sampling_mask` receive the positive sampling support
and the selected token's logprob over that support for each committed output token.
Capture stays on the device until result processing can safely consume it.

## Behavior

- Existing response fields `output_token_sampling_mask`,
  `output_token_sampling_mask_length`, and `output_token_sampling_logprobs` stay
  aligned with committed output tokens, including streaming responses.
- FlashInfer capture follows its threshold and joint-filter semantics, including
  cutoff ties. PyTorch capture uses its actual filtered, sorted weights. The
  selected-token logprob is `log(selected_weight / positive_support_mass)`.
- Greedy sampling returns the selected token as a singleton with logprob zero.
  Capturing a mask does not change seeded sampling. If token synchronization
  replaces the sampled token, its weight is read from the captured support.
- Only opted-in rows are captured, in batch order. Filtering and merging batches
  preserve the correspondence between request flags and device row indices.

## Data flow

```text
request flags -> device row indices -> sampler capture
                                       |
                          bounded IDs, lengths, logprobs, statuses
                                       |
                          result copy / pipeline transport
                                       |
                          completed copy -> CPU materialization
                                       |
                          status check -> commit token and metadata
```

The tensor result participates in the existing asynchronous device-to-host copy
and stream-lifetime protection. Overlap result processing waits for that copy
before building Python response data. Pipeline stages relay the tensor result
even when ordinary logprobs were not requested. Disaggregated prefill validates
the result before transferring the first token and mask to decode.
Pipeline results that omit logits and request no sampling metadata still commit
their output tokens normally.

## Capacity and failures

`--sampling-mask-max-tokens` defaults to `4096` and must be positive. Prefill and
decode nodes use the same capacity. Requests require greedy `top_k=1` or finite
`1 < top_k <= capacity` after sampling-parameter normalization.

Realized support can exceed `top_k` because of cutoff ties. Support exceeding
capacity aborts the request with a bad-request error; it is never returned as a
truncated mask. Invalid or missing captured support produces an internal error.
Tensor-parallel and context-parallel replicas agree on failure status. A failed
step does not commit its sampled token; request resources are released.

## Evidence

Unit, GPU capture, and server integration checks pass; see
[reproducible validation and limits](evidence/feature-00-sampling-mask.md).

## Boundaries

Sampling masks support FlashInfer and PyTorch, ordinary and overlap scheduling,
pipeline parallelism, and disaggregated serving within the server's existing
mode-combination constraints. Speculative decoding, the Ascend sampler, and
RL on-policy targets do not support sampling-mask requests. Support ordering is
backend-dependent; the mask describes a set of token IDs.
