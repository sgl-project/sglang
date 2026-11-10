# Request-level frontend wall time

`sglang:frontend_request_stage_seconds{stage="total|text|multimodal"}` records
each logical request once on its first submission to the Scheduler. Enable with
`--enable-metrics`. `total` records every submitted request; `text` and
`multimodal` record only requests that executed the stage. An executed stage with
zero measured duration is still recorded.

- `total`: HTTP route handler entry, before body parsing, until the frontend send
  returns. Excludes routing before the handler and does not prove Scheduler receipt.
- `text`: union of explicit encoding-call intervals for this request. Excludes
  queue waits and separate template rendering; inseparable template-and-encode
  calls include rendering. Shared batch execution is attributed in full to each
  participant. Cache hits and pre-tokenized inputs do not add encoding intervals.
- `multimodal`: the complete awaited local processor operation, including internal
  waiting and nested text encoding. Base/Qwen explicit encoding is also recorded
  in `text`; opaque fused processor calls are not artificially split.

Stages may overlap and cannot be added to infer total elapsed time. Histogram
quantiles describe different requests and must not be added either. Internal
parallel-sampling cache warmups do not add request samples.

Stage counts are participating request counts. Stage averages and quantiles
describe those requests, not all submitted requests. Failed or cancelled stages
retain their elapsed spans, but a request that never reaches submission does not
export any of these samples.

Use this histogram's sum/count for averages. For P50/P90/P99, aggregate buckets
by model and stage before `histogram_quantile(0.5|0.9|0.99, ...)`. An absent
metric or a window without requests should not be displayed as a synthetic zero.
Existing TTFT can start at a later boundary and is observed on first-token
arrival, so being below TTFT is not a proof of correct instrumentation.
