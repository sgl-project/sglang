## SGLang on Hathora: PD Disaggregation + EAGLE (Speculative Decoding)

This guide documents a clean, production-focused setup for using SGLang on Hathora with:
- Prefill/Decode (PD) disaggregation on H100 nodes
- EAGLE speculative decoding for higher throughput


### Defaults

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Port: `8000`
- Region: set per deployment (example: `seattle`)
- PD disaggregation: enabled automatically when exactly 2, 4, or 8 H100s are present and `TP_SIZE` is not set
- EAGLE: opt-in via environment variables


### Environment

Minimal environment you should set for Hathora:

```bash
# Networking
export HATHORA_DEFAULT_PORT="8000"
export HATHORA_REGION="seattle"

# Model
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export LOG_LEVEL="INFO"

# PD disaggregation (auto-gated; see below)
export ENABLE_PD_DEFAULTS="true"         # default=true; only takes effect on H100 with 2/4/8 GPUs and no TP_SIZE
export DISAGG_BOOTSTRAP_PORT="8998"      # optional override; default 8998
# Optional (pin IB device list if auto-detection doesn’t match your fabric)
# export DISAGG_IB_DEVICE="mlx5_0,mlx5_1,mlx5_2,mlx5_3"

# Unified mode (disable PD) options
# export ENABLE_PD_DEFAULTS="false"       # force unified
# export TP_SIZE="1"                      # any explicit TP disables PD auto split

# EAGLE speculative decoding (recommended settings; tune per model)
export SPECULATIVE_ALGORITHM="EAGLE"
export SPECULATIVE_NUM_STEPS="2"         # 1–2; higher can increase throughput but may affect TTFT
export SPECULATIVE_EAGLE_TOPK="6"        # 4–8 typical starting range
export SPECULATIVE_NUM_DRAFT_TOKENS="10" # 8–12 typical starting range
# Optional
# export SPECULATIVE_TOKEN_MAP="/path/to/token_map.json"
# export SPECULATIVE_ATTENTION_MODE="prefill"  # default
```


### PD Disaggregation: How It Works

When enabled, the server launches separate engines for prefill and decode with non-overlapping GPUs and transfers KV across them. This is auto-enabled only if:
- All visible GPUs are H100, and
- Exactly 2, 4, or 8 GPUs are available, and
- `TP_SIZE` is not set.

Splits:
- 8 GPUs: prefill `TP=2` (GPUs 0–1), decode as two replicas `TP=3` each (GPUs 2–4 and 5–7), with round-robin across the two decode replicas.
- 4 GPUs: prefill `TP=1` (GPU 0), decode `TP=3` (GPUs 1–3).
- 2 GPUs: prefill `TP=1` (GPU 0), decode `TP=1` (GPU 1).

Notes:
- To fall back to a single unified engine, either set `TP_SIZE` explicitly or set `ENABLE_PD_DEFAULTS=false`.
- `DISAGG_BOOTSTRAP_PORT` can be changed if 8998 is occupied.
- If your RDMA fabric requires pinning devices, set `DISAGG_IB_DEVICE` (e.g., `mlx5_0,mlx5_1`).


### EAGLE: Recommended Starting Point

Enable and tune EAGLE with the environment variables above. A solid starting point is:

```bash
export SPECULATIVE_ALGORITHM="EAGLE"
export SPECULATIVE_NUM_STEPS="2"
export SPECULATIVE_EAGLE_TOPK="6"
export SPECULATIVE_NUM_DRAFT_TOKENS="10"
```

Then monitor acceptance rate and latency to adjust. The ideal values depend on workload and model.


### Running on Hathora

Build and run the container using the provided Dockerfile and entrypoint:

```bash
# Build
docker build -f Dockerfile -t sglang-hathora .

# Run (locally)
docker run --gpus all --rm -p 8000:8000 \
  -e HATHORA_DEFAULT_PORT="8000" \
  -e HATHORA_REGION="seattle" \
  -e MODEL_PATH="Qwen/Qwen2.5-7B-Instruct" \
  -e ENABLE_PD_DEFAULTS="true" \
  -e SPECULATIVE_ALGORITHM="EAGLE" \
  -e SPECULATIVE_NUM_STEPS="2" \
  -e SPECULATIVE_EAGLE_TOPK="6" \
  -e SPECULATIVE_NUM_DRAFT_TOKENS="10" \
  sglang-hathora
```

Endpoints:
- Health: `GET /health` (reports `pd_mode` and `pd_bootstrap_port`)
- OpenAI chat: `POST /v1/chat/completions`


### Operational Tips

- PD is auto-gated to H100 nodes; on non-H100 or different GPU counts, unified mode is used.
- To debug PD connectivity, verify `/health`, ensure the bootstrap port is accessible, and consider setting `DISAGG_IB_DEVICE`.
- To temporarily force unified mode during troubleshooting, set `ENABLE_PD_DEFAULTS=false` or set `TP_SIZE` explicitly.
