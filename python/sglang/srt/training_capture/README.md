# Training Snapshot Producer

This package adds opt-in capture of online target-model training samples. A
sample contains token IDs, a response loss mask, selected target KV layers, and
the raw top-128 teacher logits with their vocabulary IDs and log-sum-exp. SGLang
writes tensor payloads to Mooncake and publishes immutable sample metadata
through a Catalog service.

Capture is disabled unless `--training-capture-config` is provided. The normal
serving path does not initialize capture resources when the option is absent.

## Configuration

Pass a JSON file when starting the server:

```bash
python -m sglang.launch_server \
  --model-path /models/Qwen3.5-4B \
  --training-capture-config /etc/sglang/training-capture.json
```

Minimal configuration:

```json
{
  "dataset_id": "draft-training-v1",
  "model_id": "Qwen3.5-4B",
  "producer_revision": "model-sha256",
  "selected_layer_ids": [8, 16, 24],
  "catalog_endpoint": "http://catalog.service:8080",
  "journal_directory": "/var/lib/sglang/training-capture",
  "sample_ratio": 0.01,
  "max_sample_tokens": 8192,
  "max_inflight_samples": 4,
  "max_host_bytes": 536870912,
  "storage_chunk_tokens": 256,
  "store": {
    "local_hostname": "serving-node-0",
    "master_server_addr": "mooncake-master.service:50051",
    "protocol": "rdma",
    "metadata_server": "P2PHANDSHAKE",
    "local_buffer_size": 16777216,
    "rdma_devices": "mlx5_0"
  }
}
```

The configuration is strict: unknown fields, duplicate layer IDs, relative
journal paths, invalid limits, and unsupported runtime combinations fail before
capture allocates KV exporters or registered Host memory. Optional fields cover
expected weight/tokenizer revisions, adaptive admission, device staging,
HiCache KV export, FlashInfer teacher top-k selection, payload hashing,
replication, and Catalog retry/lease limits. See `config.py` for defaults.

The Catalog bearer token can be read from the environment variable named by
`catalog_token_env`. Do not put the token in the JSON file.

Current validation rejects DP/context parallelism, non-DSpark speculative
algorithms, simulated acceptance, unsupported P/D transports, PDMux, diffusion
models, LoRA, quantized weights, custom weight/forward hooks, and embedding or
encoder-only serving. P/D capture requires the Mooncake transfer backend.

## Data Contract

The versioned JSON schemas and examples live in `schemas/`:

- `manifest.schema.json` and `manifest.example.json` describe a published
  sample and every tensor object.
- `catalog-capabilities.schema.json` and its example describe producer/consumer
  capability negotiation.
- `dspark-target-kv.schema.json` describes the target-KV draft contract.

The default contract ID is `maas-target-kv-top128-v1`, schema version 1. Tensor
descriptors include dtype, shape, byte length, digest, owner rank, Mooncake key,
and token range. Manifests identify the model, tokenizer, target weights,
selected layers, TP/PP ownership, generation, and request sample.

The logical tensors are:

- input and generated token IDs;
- a loss mask with prompt positions set to zero and response positions set to
  one;
- top-128 raw teacher logits and matching vocabulary token IDs for each response
  token;
- teacher log-sum-exp values, which let consumers reconstruct normalized
  top-k probabilities without a target-model prefill;
- K and V tensors for each configured target layer and canonical TP/PP owner.

KV remains partitioned by its native serving ownership. The manifest records
enough topology and codec information for a consumer to reconstruct the selected
global layers without assuming replicated heads or a specific page layout.

## Publication Lifecycle

Admission is deterministic from the configured seed and request identity. It is
bounded by the sample ratio, token limit, Host/device budgets, available slots,
and optional adaptive latency control. Capture never evicts serving KV to make
room for a sample.

The inference thread records completed CUDA work into request-owned contexts.
Background writers then:

1. wait for the captured copies;
2. validate tensor coverage and finite teacher values;
3. register the immutable object set with the Catalog;
4. write payload chunks through registered Mooncake buffers;
5. seal the exact manifest and owner receipts;
6. publish the sample atomically;
7. remove the durable journal entry after publication is confirmed.

Catalog fencing and idempotency keys prevent an expired or retried producer from
publishing a second generation under the same identity. Ambiguous write, seal,
or publish responses enter reconciliation; they are not reported as clean
failures until object and Catalog state has been checked. Registered buffers
remain quarantined while ownership is uncertain.

Multi-rank capture uses a dedicated CPU process group for startup agreement,
request tickets, ownership, and receipts. Only canonical owners write each
partition. A publication becomes visible after every required owner has
submitted a matching receipt.

The Catalog API used by the producer is:

| Route | Purpose |
| --- | --- |
| `GET /capabilities` | Agree on contract, codec, Store transport, pinning, retention, and metadata limits |
| `POST /captures:begin` | Reserve a fenced capture identity and byte budget |
| `POST /captures/{id}/heartbeat` | Renew the active lease |
| `POST /captures/{id}/objects` | Register object descriptors and written receipts |
| `POST /captures/{id}/seal` | Validate and prepare the immutable manifest |
| `POST /samples:publish` | Atomically expose the sample and Catalog cursor |
| `POST /captures/{id}/fail` | Record a terminal producer failure |

The Catalog owns consumer leases, checkpoint retention, and garbage collection.
The producer does not delete a published sample after a consumer reads it.

## Runtime Control

Capture can be controlled without restarting serving:

```bash
curl -X POST http://127.0.0.1:30000/control_training_capture \
  -H 'Content-Type: application/json' \
  -d '{"action":"pause"}'

curl -X POST http://127.0.0.1:30000/control_training_capture \
  -H 'Content-Type: application/json' \
  -d '{"action":"resume"}'

curl -X POST http://127.0.0.1:30000/control_training_capture \
  -H 'Content-Type: application/json' \
  -d '{"action":"abort"}'
```

`pause` blocks new admission and lets current writers finish. `resume` reopens
admission after health checks. `abort` also fails or drains unfinished captures;
published and ambiguously published samples remain governed by Catalog state.

Online weight replacement, memory release, and other operations that could
invalidate target identity are rejected while capture owns active work.

## Metrics

With `--enable-metrics`, `/metrics` exports bounded-label
`sglang:training_capture_*` series for:

- admission, routing, capture, failure, and publication events;
- available/active/queued/writing reservations;
- free/filling/quarantined registered Host slots;
- Host and device allocation against configured budgets;
- queue depth, oldest writer age, pause/disabled/cooldown state;
- TTFT/TPOT latency-control state and observations;
- background stage calls, failures, total seconds, and maximum seconds;
- KV export bytes enqueued to Host or device staging.

Metrics are sampled by a Host background thread. They do not contain request
IDs, object keys, sample contents, or exception text. Stage durations are Host
wall time and can include scheduling waits. Enqueued KV bytes are not completed
D2H or RDMA wire bandwidth. Use Catalog state, rather than metric ratios, to
measure exact dataset completeness.

The Grafana dashboard is
`examples/monitoring/grafana/dashboards/json/training-capture-dashboard.json`.

## Trace Summary

`python -m sglang.benchmark.summarize_training_capture_trace` attributes CPU
operators, CUDA kernels, and D2H/D2D copies to `training_capture.*` profiler
ranges by CUDA correlation ID:

```bash
python -m sglang.benchmark.summarize_training_capture_trace \
  capture.trace.json.gz \
  --output capture-summary.json \
  --require-capture
```

The report separates CPU scope time from asynchronous GPU work and rejects
ambiguous overlapping scopes.

## Verification

Run the CPU-focused contract and coordinator suite:

```bash
PYTHONPATH=python python -m pytest test/registered/unit/training_capture -q
```

Mooncake integration tests require the Mooncake SDK and `mooncake_master`:

```bash
PYTHONPATH=python python -m pytest \
  test/registered/storage/test_training_snapshot_mooncake.py -q
```

The serving runtime test requires a supported model and GPU:

```bash
PYTHONPATH=python python \
  test/registered/storage/test_training_capture_runtime.py \
  --model-path /models/Qwen3.5-4B -v
```

The runtime test reads the sample from Mooncake after producer shutdown and
checks tokens, mask, selected K/V, top-128 vocabulary IDs, raw logits, and
log-sum-exp against an independent observer.
