# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

Serves a single model and routes across its workers. Exposes
`/v1/tokenize`, `/v1/detokenize`, `/v1/models`, `/v1/chat/completions`
(buffered and SSE), plus `/healthz` / `/readyz` and `/metrics`. Worker
pools come from either a static URL list or Kubernetes EndpointSlice
discovery.

## Building

```bash
cd experimental/sgl-router
cargo build --release
```

## Running

The router is configured entirely through CLI flags (run
`sgl-router --help` for the full list). It serves exactly one model, so
`--model-id` is required, along with exactly one discovery backend.
`--tokenizer-path` is optional: give it a local `tokenizer.json` path or a
HuggingFace repo id, and when omitted the router downloads the tokenizer
for `--model-id` from HuggingFace (honoring `HF_TOKEN` / `HF_HOME`).

Static worker list:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000
```

Kubernetes EndpointSlice discovery:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --service-discovery \
  --service-discovery-namespace prod \
  --selector app=engines-qwen3
```

Omit `--service-discovery-namespace` to watch all namespaces (requires
cluster-wide RBAC). For prefill/decode disaggregation, replace `--selector`
with `--prefill-selector` and `--decode-selector`.

## Token dataset export

The optional token-export tee writes gzipped NDJSON batches without blocking
request serving. A full queue or upload backlog drops export records and
increments `sgl_router_token_export_total`.

Amazon S3 uses static SigV4 credentials:

```bash
export RADIXARK_TOKEN_EXPORT_S3_URI=s3://bucket/prefix/
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-west-2
```

Google Cloud Storage uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials):

```bash
export RADIXARK_TOKEN_EXPORT_GCS_URI=gs://bucket/prefix/
```

ADC supports `GOOGLE_APPLICATION_CREDENTIALS`, local `gcloud` ADC, and the
metadata service used by GCE service accounts and GKE Workload Identity.
Production deployments should use a dedicated service account with
`roles/storage.objectCreator` on the destination bucket. Export objects contain
prompt/output token sequences and client key identifiers, so bucket readers and
retention must be restricted accordingly. S3 and GCS export are mutually
exclusive.

## License

Apache-2.0.
