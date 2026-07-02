# Request-Level PD Flip SLO Monitor Design

## Goal

Add an experiment-scoped monitor mode where each trace request carries its own
TTFT and TPOT SLO. The monitor computes SLO attainment from request-level trace
events and feeds the existing PD flip controller state machine without changing
the default Prometheus histogram monitor.

## Non-Goals

- Do not replace the existing `/metrics` histogram monitor.
- Do not add high-cardinality SLO labels to Prometheus metrics.
- Do not make streaming handoff transparent in this change.
- Do not change KV migration manifests or role-flip state transitions.

## Request Contract

Trace requests carry SLOs through the existing OpenAI-compatible
`custom_params` field:

```json
{
  "model": "deepseek_v3.1_terminus",
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 512,
  "stream": false,
  "custom_params": {
    "pd_flip_slo": {
      "ttft_seconds": 5.0,
      "tpot_seconds": 0.02
    }
  }
}
```

`ttft_seconds` is user-observed time from request send to first token arrival.
`tpot_seconds` is the per-token interval target used for token-weighted decode
attainment.

## Trace Ledger

The trace runner writes a JSONL ledger beside the trace artifacts. Each request
gets one mutable record keyed by `request_id`.

Minimum fields:

```json
{
  "request_id": "req-0001",
  "status": "running",
  "start_time": 238100.1,
  "first_token_time": null,
  "end_time": null,
  "ttft_slo_seconds": 5.0,
  "tpot_slo_seconds": 0.02,
  "ttft_seconds": null,
  "ttft_met": null,
  "good_tpot_intervals": 0,
  "total_tpot_intervals": 0,
  "last_token_time": null,
  "completion_tokens": 0,
  "error": null
}
```

For non-streaming responses, the first implementation cannot observe token
arrival times progressively from the client. It may use response-level
`_trace_elapsed_seconds` for completed requests and record no TPOT intervals
unless streaming or server-side events are enabled. The monitor must therefore
support partial information:

- TTFT samples only count when `ttft_seconds` is known.
- TPOT samples only count when interval data exists.
- Missing data does not count as good or bad.

The first Docker trace for this monitor should use streaming client collection
only for measuring TTFT/TPOT. This does not imply streaming KV handoff
transparency; the migrated transparent handoff guarantee remains non-streaming.
The streaming measurement path is a monitoring input, not the target migrated
request path.

## SLO Computation

The trace SLO monitor reads the ledger every poll and computes a sliding-window
snapshot.

TTFT attainment:

```text
prefill_slo_attainment =
  count(request.ttft_seconds <= request.ttft_slo_seconds)
  / count(requests with ttft_seconds observed)
```

TPOT attainment:

```text
decode_slo_attainment =
  sum(request.good_tpot_intervals)
  / sum(request.total_tpot_intervals)
```

This keeps TPOT token-weighted, matching the current histogram-style TPOT
semantics more closely than a request-weighted mean.

Records outside the configured window are ignored. The window uses request
event timestamps, not file modification time.

## Controller Integration

The existing controller state machine remains unchanged:

```text
safe -> preparing_kv_transfer -> flipping_role -> safe
```

The controller already depends on a `slo_monitor.collect_cluster(...)` object
that returns a `ClusterSLOSnapshot`. The trace monitor should implement the same
method and return the same snapshot shape, filling:

- `prefill_slo_attainment` from request-level TTFT attainment.
- `decode_slo_attainment` from request-level TPOT interval attainment.
- `nodes` with load data from existing worker polling when available.

If the trace ledger has no observed TTFT/TPOT samples, the corresponding
attainment remains `None`, and the controller does not trigger from that metric.

## Experiment Flow

The trace runner gains a request-level SLO mode:

1. Load or synthesize trace requests with per-request TTFT/TPOT SLO.
2. Send requests with `custom_params.pd_flip_slo`.
3. Record request events into the ledger.
4. Run controller monitor with the trace SLO monitor mode.
5. Preserve existing artifacts:
   - `monitor.json`
   - `client_response.json`
   - `summary.json`
6. Add request-SLO artifacts:
   - `trace_requests.jsonl`
   - `trace_slo_ledger.jsonl`
   - `trace_slo_summary.json`

## Safety

The default Docker harness behavior is unchanged. Request-level SLO monitoring
is opt-in through trace runner/controller flags. The runner must keep the
existing safety property: it may drain or undrain Codex-owned router workers,
but it must not stop, remove, kill, or restart containers.

## Testing

Unit tests should cover:

- Parsing request SLOs from `custom_params.pd_flip_slo`.
- TTFT attainment with heterogeneous per-request thresholds.
- TPOT token-weighted attainment with heterogeneous per-request thresholds.
- Missing TTFT/TPOT data yields `None` instead of false misses.
- Controller can consume a trace SLO monitor snapshot without changing the FSM.
- Docker runner strings include the new opt-in mode and still exclude destructive
  Docker commands.

Integration verification should run:

```text
python3 -m unittest discover -s test/srt -p 'test_pd_flip*.py'
```

Then run a safe Docker trace in the existing three-node Codex topology.

## Remaining Limitations

- Non-streaming client responses do not expose progressive token intervals, so
  TPOT measurement requires a streaming measurement request or future
  server-side event export.
- Request-level SLO monitor is an experiment mode, not a production metrics
  replacement.
- Flip direction remains simple in the first version: TTFT risk drives D->P and
  TPOT risk drives P->D.
