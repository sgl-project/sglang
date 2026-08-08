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

## Docker

Build the image from the repository root:

```bash
docker build -f docker/sgl-router.Dockerfile -t sgl-router:dev .
```

The image uses the same CLI-only configuration as the native binary. Bind the
router to `0.0.0.0` so its port is reachable outside the container, and pass a
worker URL that the container can resolve and reach:

```bash
docker run --rm -p 30000:30000 \
  sgl-router:dev \
  --host 0.0.0.0 --port 30000 \
  --model-id Qwen/Qwen3-0.6B \
  --worker-urls http://worker.example.com:30000
```

Replace `worker.example.com` with the worker's DNS name or IP address. If the
worker also runs in Docker, attach both containers to the same Docker network
and use the worker container name. Add `--tokenizer-path` and mount a local
`tokenizer.json` when the router should not download the tokenizer from
HuggingFace.

## License

Apache-2.0.
