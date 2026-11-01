# Tokenizer encode wall time

## Request-level frontend timing

`sglang:frontend_request_stage_seconds{stage="total|text|multimodal|chat_template"}`
records each logical request once when it is submitted. `total` starts at the
HTTP route handler before body parsing and ends when the frontend send returns;
it does not include routing before that handler or prove Scheduler receipt.
`text` accumulates encoding calls, including inseparable template encoding;
`chat_template` identifies that subset. `multimodal` measures the awaited
processor operation, including any internal encoding not separately timed.
Direct tokenizer calls in the base and Qwen multimodal processors also record
`text`, including when executed in the processor thread pool. These spans are
nested within `multimodal`. Fused Hugging Face processor calls are not split
into an artificial text-only duration.
Stages may overlap and their quantiles must not be added. Unused stages record
zero. Shared batch encoding is attributed in full to each participating request,
and internal parallel-sampling cache warmups do not create request samples.

Use the histogram's own sum/count for averages. For P50/P90/P99, aggregate its
buckets by model and stage before `histogram_quantile(0.5|0.9|0.99, ...)`.
Existing TTFT starts at a later serving boundary on some endpoints, and records
a different population, so comparisons need a drained, verified replay window.

## Encoding-call timing

`sglang:tokenizer_encode_seconds` measures prompt encoding with
`time.perf_counter()`. It is independent of the process CPU counter and uses its
own histogram count as the denominator. Enable it with `--enable-metrics`.

One observation represents one measured encoding operation, not one HTTP request,
generated sequence, token, or CPU thread. A batch operation is recorded once with
its entire elapsed wall time; its duration is never divided by batch size. A
request can perform several encoding operations, such as encoding an assistant
prefix separately. Cached encodings and pre-tokenized `input_ids` do not produce
zero-duration observations.

## Dimensions and boundaries

| Label | Values | Meaning |
| --- | --- | --- |
| `source` | `chat`, `manager` | Chat serving code, or TokenizerManager paths including `/generate`, completions and embeddings. |
| `input_type` | `single`, `batch` | One logical input or several inputs in one measured operation. |
| `operation` | `encode`, `chat_template` | Encoding only, or a fused chat renderer/encoder that cannot safely be split. |
| `result` | `success`, `error` | Whether the measured operation returned or raised. |

The metric also inherits the TokenizerMetricsCollector's base model/engine
labels. The default dashboard selects successful, single-input `encode`
operations and draws `chat` and `manager` separately. It does not mix batches or
fused template operations into that population.

Normal Chat encoding excludes separate template rendering, template-cache key
construction, and decoding cached prompt text. The `chat_template` operation
explicitly includes fused rendering and encoding for compatibility paths.
Asynchronous dynamic batching starts its timer **inside the executor thread**,
so collection and executor queue waits are excluded. All paths preserve the
original outputs and exceptions. Startup probes and auxiliary encodings used to
configure response parsers are outside this prompt-encoding metric.

## Dashboard

Import `dashboard.json` as a **new dashboard** and select a Prometheus datasource.
Its ID and UID are unset. It shows the five-minute call-weighted average,
P50/P90/P99, successful call rate and failed call rate. Change `input_type` or
`operation` explicitly to inspect another population. No traffic or missing
instrumentation is shown as missing data, not a synthetic zero.

The average is calculated from the same histogram's sum and count:

```promql
sum by (model_name, source) (
  rate(sglang:tokenizer_encode_seconds_sum{input_type="single",operation="encode",result="success"}[5m])
)
/
sum by (model_name, source) (
  rate(sglang:tokenizer_encode_seconds_count{input_type="single",operation="encode",result="success"}[5m])
)
```

For a percentile, aggregate histogram buckets across instances before calling
`histogram_quantile`; do not average per-instance percentiles. Classic Prometheus
uses the `_bucket` series. Prometheus-compatible storage may rewrite histogram
names: adapt only the bucket selector to its documented convention, keeping the
sum, count, filters and grouping consistent.

Quantiles are bucket estimates, not exact measurements. The first finite bucket
is 1 ms, so submillisecond quantiles have limited resolution; the average still
uses the recorded durations directly. Multimodal processor encodings outside the
instrumented Chat/TokenizerManager paths are not covered by this first version.

The existing `sglang:process_cpu_seconds_total{component="tokenizer"}` remains a
process resource metric. Its rate measures CPU cores consumed. Dividing it by
TTFT sample rate does not produce tokenizer wall time. This new histogram is
also not a full TokenizerManager-stage or TTFT critical-path measurement, and
its ratio to TTFT must not be interpreted as a predicted optimization benefit.

## Verification

CPU tests cover Chat rendering/cache boundaries, whole-batch accounting,
preserved exceptions, disabled metrics, executor queue exclusion, fused template
separation, and histogram aggregation from independent processes without a
Scheduler:

```bash
PYTHONPATH=python python test/registered/unit/observability/test_tokenizer_encode_metrics.py
```
