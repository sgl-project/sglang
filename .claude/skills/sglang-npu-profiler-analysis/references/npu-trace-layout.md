# NPU Trace Layout

After a successful SGLang NPU profile capture, directories typically look like:

```text
<output_dir>/
  <timestamp>/                     # present when using bench_serving --profile
    <hostname>_<pid>_<ts>_ascend_pt/
      ASCEND_PROFILER_OUTPUT/
        trace_view.json
        analysis.db
        ascend_pytorch_profiler_0.db
        kernel_details.csv
        operator_details.csv
        step_trace_time.csv
```

## Key files

| File | Contents |
| --- | --- |
| `operator_details.csv` | Operator name, duration, device time — primary ranking input |
| `kernel_details.csv` | Lower-level kernel entries |
| `trace_view.json` | Chrome trace / MindStudio Insight timeline |
| `step_trace_time.csv` | Step boundaries for prefill/decode separation |

## Server log anchors

Search logs for:

- `Profiling starts. Traces will be saved to:`
- `Profiling done. Traces are saved to:`

Those lines are authoritative when multiple timestamp folders exist.
