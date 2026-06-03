# NVFP4 KV Cache on SM120 (Blackwell)

NVFP4 KV cache uses FP4 (E2M1) format with two-level scaling (per-tensor FP32 + per-block FP8 E4M3) to compress the KV cache on Blackwell GPUs. This reduces KV cache memory by ~2x compared to FP8, enabling longer context or higher throughput.

## Supported Hardware

NVFP4 KV cache requires Blackwell SM100 or SM120 architecture. This guide focuses on SM120.

## Installation

### Prerequisites

- CUDA 13.0+
- PyTorch 2.9.1+
- FlashInfer >= 0.6.3 (latest main branch)
- SM120 GPU

### Install SGLang

NVFP4 KV cache support is on the `nvfp4-kvcache-sm120-v2` branch. Clone from the fork and install in editable mode:

```bash
git clone -b nvfp4-kvcache-sm120-v2 https://github.com/samuellees/sglang.git
cd sglang
pip install -e "python[all]"
```

See [SGLang Installation Guide](https://docs.sglang.io/get_started/install.html) for more installation options.

### Install FlashInfer from Source

See [FlashInfer Installation Guide](https://docs.flashinfer.ai/installation.html) for full details.

SGLang's `pip install` pulls in a pre-built FlashInfer wheel, but NVFP4 KV cache requires the latest main branch built from source. After installing SGLang, uninstall the pre-built packages and rebuild:

```bash
# 1. Uninstall pre-built FlashInfer packages
pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache

# 2. Clone and build FlashInfer from source
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install --upgrade pip setuptools
pip install -v .
```

> **Note**: Building from source requires CUDA 13.0+ and may take 10-20 minutes. The `--recursive` flag is required to fetch submodules.

## Usage

### Basic: FP4 KV Cache

```bash
python3 -m sglang.launch_server \
    --model-path <model_path> \
    --tp-size <num_gpus> \
    --kv-cache-dtype nvfp4 \
    --prefill-attention-backend flashinfer \
    --decode-attention-backend trtllm_mha \
    --moe-runner-backend triton \
    --mamba-ssm-dtype bfloat16 \
    --disable-radix-cache \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 \
    --mem-fraction-static 0.6 \
    --host 0.0.0.0 --port 30000
```

Key arguments:
- `--kv-cache-dtype nvfp4`: Enable NVFP4 KV cache
- `--prefill-attention-backend flashinfer`: Use FlashInfer for prefill (dequantizes FP4→FP8 for prefill kernel)
- `--decode-attention-backend trtllm_mha`: Use TRT-LLM XQA decode kernel (native FP4 support)
- `--moe-runner-backend triton`: Use Triton MoE runner
- `--mamba-ssm-dtype bfloat16`: Use BF16 for Mamba SSM (for hybrid models like Qwen3.5)

### Speculative Decoding / MTP

Speculative decoding and MTP-specific FP4 KV paths are intentionally out of scope
for this version. Keep `--speculative-algorithm` unset when validating the
functional NVFP4 KV cache path.

## Testing

### GSM8K Accuracy Check

Start the server with NVFP4 KV cache, then run:

```bash
# Full evaluation (1319 questions, recommended for accurate results)
python3 benchmark/gsm8k/bench_sglang.py \
    --num-questions 1319 \
    --parallel 1319 \
    --port 30000 \
    --max-new-tokens 10240
```

Recommended model: [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

Reference results (Qwen3.5-35B-A3B, 4x RTX PRO 6000 Blackwell SM120, TP=4):

#### GSM8K (1319 questions, max_tokens=10240)

| KV Cache | MTP | Accuracy | Throughput |
|----------|-----|----------|------------|
| BF16 | No | 91.3% | - |
| FP8 (fp8_e4m3) | No | 91.0% | 350.6 tok/s |
| **NVFP4** | **No** | **91.4%** | **2528 tok/s** |

#### GPQA (198 questions, 8 repeats, temperature=0.6, max_tokens=81920)

| KV Cache | Mean Score | Scores (8 rounds) |
|----------|------------|-------------------|
| BF16 | 83.5% | 84.3, 83.8, 83.8, 84.3, 80.8, 85.4, 82.3, 82.8 |
| FP8 (fp8_e4m3) | 82.1% | 82.8, 80.3, 85.4, 81.3, 80.8, 82.3, 80.8, 82.8 |
| **NVFP4** | **80.1%** | 81.8, 80.3, 79.3, 81.3, 80.3, 80.8, 79.3, 77.8 |

#### LongBench V2 (503 questions, 128K context, no thinking, max_tokens=16384)

| KV Cache | Score | Easy | Hard | Latency |
|----------|-------|------|------|---------|
| BF16 (3-round avg) | 52.4% +/- 0.3% | 56.3% | 50.2% | 1146s |
| **NVFP4** | **49.7%** | 55.7% | 46.3% | 1059s |
| FP4 (3-round avg, old) | 48.9% +/- 0.5% | 52.8% | 46.7% | 1471s |

#### AIME25 (Majority Vote, n=16 beams, max_tokens=114688)

| KV Cache | Majority Vote | Pass@16 | Problems |
|----------|---------------|---------|----------|
| BF16 (B200 x2) | 93.3% (28/30) | 87.9% (422/480) | 30/30 |
| **FP4 (RTX 6000 x4)** | **(pending retest)** | - | - |

#### Summary

| Benchmark | FP4 | BF16 | FP8 | Delta (vs BF16) |
|-----------|-----|------|-----|-----------------|
| GSM8K | 91.4% | 91.3% | 91.0% | +0.1pp |
| GPQA | 80.1% | 83.5% | 82.1% | -3.4pp |
| LongBenchV2 128K | 49.7% | 52.4% | - | -2.7pp |
| AIME25 MV | (pending) | 93.3% | - | - |

### GPQA Accuracy Check

```bash
python3 -m sglang.test.run_eval --eval-name gpqa --port 30000 \
  --model <model_path> --num-examples 198 --max-tokens 81920 \
  --repeat 8 --temperature 0.6 --top-p 0.95 --top-k 20 --num-threads 16
```

### LongBench V2 Accuracy Check

```bash
# Start server with 128K context settings:
# --chunked-prefill-size 32768 --max-prefill-tokens 65536
# --max-running-requests 128 --mem-fraction-static 0.75
# --cuda-graph-bs 1 2 4 8 16 32 64 128

python3 -m sglang.test.run_eval --eval-name longbench_v2 --port 30000 \
  --model <model_path> --max-context-length 128000 --max-tokens 16384 \
  --num-threads 16 --chat-template-kwargs '{"enable_thinking": false}'
```

## Architecture

### Two-Level Scaling

NVFP4 uses a two-level quantization scheme:
1. **Per-tensor scale** (FP32): Global scale factor for the entire KV cache tensor
2. **Per-block scale** (FP8 E4M3): Fine-grained scale per block of elements

These scales are stored in `kv_cache_sf` (scale factor) tensors alongside the FP4 KV data.

### Kernel Dispatch

- **Prefill**: FlashInfer dequantizes FP4→FP8 on-the-fly, then uses the standard FP8 prefill kernel
- **Decode**: TRT-LLM XQA kernel reads FP4 data natively with the two-level scales

### Changed Files (vs main)

| File | Change |
|------|--------|
| `server_args.py` / `model_runner.py` | Expose explicit `--kv-cache-dtype nvfp4` and map it to FP4 E2M1 storage |
| `kv_cache_quant_method.py` | Own FP4 KV recipe selection, buffer creation, and quant/dequant strategy |
| `kvfp4_tensor.py` | Provide low-level MXFP4/NVFP4 tensor helper operations |
| `memory_pool.py` | Store packed FP4 KV, per-block scales, and shared dequant workspaces |
| `flashinfer_backend.py` | Use NVFP4 dequant workspace for prefill/cache reuse |
| `trtllm_mha_backend.py` | Enable NVFP4 decode-only XQA path |

## Known Limitations

- NVFP4 KV cache requires Blackwell SM100 or SM120; it is not supported on SM90.
- TRTLLM MHA with NVFP4 KV cache is decode-only; use `flashinfer` or `triton` for prefill.
- Speculative decoding / MTP-specific FP4 KV paths are not included in this version.
- FlashInfer prefill with NVFP4 dequantizes to FP8, not native FP4 prefill
- Radix cache must be disabled (`--disable-radix-cache`)
