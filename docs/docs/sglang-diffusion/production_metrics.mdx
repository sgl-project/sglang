---
title: "Production Metrics"
description: "Monitor SGLang Diffusion requests, queues, stages and LoRA state with Prometheus."
---

## Enable metrics

Enable metrics on an NVIDIA CUDA deployment, for example:

```bash
sglang serve --model-path black-forest-labs/FLUX.2-klein-4B \
  --num-gpus 1 --enable-metrics --port 30000
curl http://localhost:30000/metrics
```

Metrics are opt-in. Disabled metrics do not scan queues or collect LoRA status.
Enabled metrics add host-side bookkeeping, not GPU synchronization or collectives.

## Metric reference

All diffusion metrics carry `role` and `replica` labels. `replica` is the
scheduler endpoint; only each DP replica's leader publishes, so TP/SP ranks do
not multiply request counts. The table lists additional labels.

| Metric | Type | Labels | Description |
| --- | --- | --- | --- |
| `sglang:diffusion_num_queue_reqs` | Gauge | none | Original generation requests waiting for their first dispatch. |
| `sglang:diffusion_num_running_reqs` | Gauge | none | Number of diffusion generation requests dispatched by the scheduler and not yet finished. |
| `sglang:diffusion_requests_total` | Counter | `status`, `is_warmup` | Completed diffusion generation requests. Dynamic batches are counted per original scheduler request after the merged output is split. |
| `sglang:diffusion_request_latency_seconds` | Histogram | `status`, `is_warmup` | Scheduler acceptance to completion, excluding HTTP preprocessing, media encoding and response delivery. |
| `sglang:diffusion_queue_time_seconds` | Histogram | `is_warmup` | Time spent waiting in the diffusion scheduler queue. |
| `sglang:diffusion_generation_batch_size` | Histogram | `stop_reason` | Generation batch size selected by the diffusion scheduler at dispatch time. |
| `sglang:diffusion_stage_host_latency_seconds` | Histogram | `stage` | Host wall time around a stage, not GPU kernel execution time. Step labels are normalized to `DenoisingStep`. |
| `sglang:diffusion_lora_loaded_adapters` | Gauge | none | Number of loaded diffusion LoRA adapters. |
| `sglang:diffusion_lora_active_modules` | Gauge | none | Number of diffusion modules with active LoRA adapters. |
| `sglang:diffusion_lora_active_adapters` | Gauge | none | Number of unique active diffusion LoRA adapters. |
| `sglang:diffusion_lora_module_active` | Gauge | `module` | Whether a diffusion module currently has an active LoRA adapter. |

Request counts refer to original scheduler requests, not generated images,
denoising steps or distributed shards. `status` is `success` or `error`;
`is_warmup` is `true` or `false`. Queue and running gauges include warmup.
Stage observations include warmup and count stage invocations, not requests;
asynchronous GPU work can complete in a later stage. For synchronized diagnostic
timings, use `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1` separately, accepting its
synchronization overhead. LoRA gauges update at startup and after LoRA control operations.

## Disaggregated serving

In disaggregated serving, the head (`role="server"`) records the original request
lifecycle, including role handoffs and errors. Queue time ends at the first
encoder dispatch; intermediate waits remain part of request latency. Workers
report their own stage and LoRA metrics, not duplicate completed requests.
The generation-batch histogram currently describes monolithic scheduling only.

Single-host pool mode exposes all child metrics through the head's `/metrics`.
A scrape aggregates only processes sharing that host's metrics directory.
For standalone remote roles, pass `--enable-metrics` to each process and scrape
its `--host`/`--port` as well as the head. Each role serves a metrics-only HTTP
endpoint. Use a separate, empty `PROMETHEUS_MULTIPROC_DIR` per server launch if
you set it yourself; otherwise SGLang creates and owns a temporary directory.
Do not share this directory between independent servers or reuse stale files.

## Query throughput

Successful, non-warmup request throughput:

```promql
sum(rate(sglang:diffusion_requests_total{status="success",is_warmup="false"}[5m]))
```
