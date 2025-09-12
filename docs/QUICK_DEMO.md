## Quick Demo: SGLang H100 Serving Container

This is a minimal end-to-end demo for running the SGLang serving container, selecting a model, and verifying health/metrics/inference.

Prereqs: H100 GPU node (or A100 for local smoke test), Docker, CUDA/NVIDIA drivers, Internet access to Hugging Face.

### 1) Build the image

```bash
cd /path/to/sglang
# If you have a specific Dockerfile for this image, use it (as referenced elsewhere):
docker build -f Dockerfile.hathora -t sglang-hathora:latest .
# Otherwise, use your existing Dockerfile that installs python deps and copies entrypoint.sh + serve_hathora.py
```

### 2) Choose one of the config styles

Option A: DEPLOYMENT_CONFIG_JSON (recommended)

Public model (no HF token required):
```bash
export DEPLOYMENT_CONFIG_JSON='{
  "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
  "tp_size": 2,
  "enable_metrics": true,
  "h100_only": true
}'
```

Gated model (HF token required):
```bash
export DEPLOYMENT_CONFIG_JSON='{
  "hf_token": "hf_xxx",
  "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "tp_size": 2,
  "enable_metrics": true,
  "h100_only": true
}'
```

Option B: Individual envs (if you prefer)
```bash
export MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3
export TP_SIZE=2
export ENABLE_METRICS=true
export H100_ONLY=true
# export HF_TOKEN=hf_xxx   # only for gated models
```

### 3) Run the container

```bash
docker run --rm -p 8000:8000 \
  --gpus all \
  -e DEPLOYMENT_CONFIG_JSON \
  -e HATHORA_DEFAULT_PORT=8000 \
  -e HATHORA_REGION=local \
  sglang-hathora:latest
```

Notes:
- H100-only is enforced when `h100_only=true`; the container will exit on non-H100 GPUs.
- To use 8x H100 on a single node, set `tp_size` to 8 (and ensure 8 GPUs are available).

### 4) Verify health and metrics

```bash
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/metrics | head -n 20
```

### 5) Send an inference request (OpenAI-compatible)

Non-streaming:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sglang",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 64,
    "stream": false
  }' | jq
```

Streaming (SSE):
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sglang",
    "messages": [{"role": "user", "content": "Tell me a short joke"}],
    "max_tokens": 64,
    "stream": true
  }'
```

### 6) Optional: Autoscaling hints

Provide targets for your external autoscaler via config:
```bash
export DEPLOYMENT_CONFIG_JSON='{
  "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
  "tp_size": 2,
  "enable_metrics": true,
  "autoscale_target_queue_depth": 400,
  "max_queued_requests": 4096
}'
```

Your autoscaler should scrape `/metrics` (Prometheus) for `sglang:num_queue_reqs`, `sglang:gen_throughput`, latency histograms, and optionally DCGM GPU metrics, then scale via your control plane.


