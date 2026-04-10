# Auto-Spec Benchmark

Benchmark toolkit for **auto-spec** (adaptive speculative decoding), which dynamically adjusts the number of speculative decoding steps at runtime based on acceptance rate and batch size.

## Overview

- `prepare_mix_dataset.py` -- Downloads and merges GSM8K, HumanEval, MT-Bench, ShareGPT (and optionally HotpotQA, SQuAD, DROP, MBPP) into a single JSONL file.
- `bench_auto_spec.py` -- Self-contained HTTP benchmark that sends concurrent requests to `/v1/completions`, collects throughput, latency, and TTFT.

## Quick Start

### 1. Prepare Dataset

The script auto-downloads small datasets (GSM8K, HumanEval, MT-Bench, MBPP, SQuAD). For large datasets (ShareGPT, HotpotQA), you may need to download them manually first (see [Manual Download](#manual-download) below).

```bash
# Quick start: base datasets only (auto-downloads ~5MB total, no ShareGPT)
python benchmark/auto_spec/prepare_mix_dataset.py \
    --output mix_spec_dataset.jsonl \
    --datasets gsm8k,humaneval,mtbench

# Full base datasets including ShareGPT (requires manual download, see below)
python benchmark/auto_spec/prepare_mix_dataset.py \
    --output mix_spec_dataset.jsonl \
    --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json

# All 8 datasets (base + medium-length)
python benchmark/auto_spec/prepare_mix_dataset.py \
    --output mix_spec_dataset.jsonl \
    --include-medium-length \
    --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --hotpotqa-path /path/to/hotpot_dev_fullwiki_v1.json
```

If a download fails, the script skips that dataset and continues with the rest.

### Manual Download

Some datasets are large or hosted on slow servers. Download them manually if auto-download fails:

**ShareGPT** (~400MB) -- Required for realistic multi-turn conversation prompts:
```bash
# Option 1: wget
wget -O ShareGPT_V3_unfiltered_cleaned_split.json \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Option 2: huggingface-cli (if installed)
huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
    --local-dir .
```

**HotpotQA** (~45MB) -- Multi-hop reasoning QA with long contexts:
```bash
wget -O hotpot_dev_fullwiki_v1.json \
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
```

Then pass the local paths:
```bash
python benchmark/auto_spec/prepare_mix_dataset.py \
    --output mix_spec_dataset.jsonl \
    --sharegpt-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --hotpotqa-path ./hotpot_dev_fullwiki_v1.json \
    --include-medium-length
```

### 2. Start SGLang Server with Auto-Spec

```bash
# Qwen3-235B-A22B (SpecV2 + auto-spec)
SGLANG_ENABLE_SPEC_V2=True python -m sglang.launch_server \
    --model Qwen/Qwen3-235B-A22B \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path /path/to/eagle-model \
    --speculative-num-steps 3 \
    --auto-spec \
    --port 30000 \
    --tp 4

# GLM-4.7-FP8 (SpecV2 + auto-spec)
SGLANG_ENABLE_SPEC_V2=True python -m sglang.launch_server \
    --model /path/to/glm-4-9b-chat-hf-fp8 \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path /path/to/eagle-model \
    --speculative-num-steps 3 \
    --auto-spec \
    --port 30000 \
    --tp 1
```

### 3. Run Benchmark

```bash
# Basic run
python benchmark/auto_spec/bench_auto_spec.py \
    --port 30000 \
    --dataset-path mix_spec_dataset.jsonl \
    --max-concurrency 16

# Sweep over concurrency levels (simulates different batch sizes)
python benchmark/auto_spec/bench_auto_spec.py \
    --port 30000 \
    --dataset-path mix_spec_dataset.jsonl \
    --batch-sizes 1,4,8,16,32 \
    --num-prompts-each 32

# Filter to specific source datasets
python benchmark/auto_spec/bench_auto_spec.py \
    --port 30000 \
    --dataset-path mix_spec_dataset.jsonl \
    --source-filter gsm8k,humaneval \
    --num-prompts 50

# Fixed output length
python benchmark/auto_spec/bench_auto_spec.py \
    --port 30000 \
    --dataset-path mix_spec_dataset.jsonl \
    --output-len 256
```

## Arguments

### prepare_mix_dataset.py

| Argument | Description |
|---|---|
| `--output` | Output JSONL path (default: `./mix_spec_dataset.jsonl`) |
| `--num-gsm8k` | Number of GSM8K samples (default: all ~1319) |
| `--num-humaneval` | Number of HumanEval samples (default: all 164) |
| `--num-mtbench` | Number of MT-Bench samples (default: all 80) |
| `--num-sharegpt` | Number of ShareGPT samples (default: all) |
| `--include-medium-length` | Also include HotpotQA, SQuAD, DROP, MBPP |
| `--datasets` | Comma-separated list to select specific datasets |
| `--sharegpt-path` | Path to locally downloaded ShareGPT JSON file |
| `--hotpotqa-path` | Path to locally downloaded HotpotQA JSON file |
| `--cache-dir` | Directory to cache downloaded datasets |

### bench_auto_spec.py

| Argument | Description |
|---|---|
| `--host` | Server host (default: `127.0.0.1`) |
| `--port` | Server port (default: `30000`) |
| `--dataset-path` | Path to mix-spec JSONL dataset |
| `--num-prompts` | Total number of prompts (default: all) |
| `--max-concurrency` | Max concurrent requests (default: 16) |
| `--output-len` | Fixed output length for all requests |
| `--source-filter` | Comma-separated source filter (e.g., `gsm8k,humaneval`) |
| `--num-prompts-each` | Number of prompts per source dataset |
| `--batch-sizes` | Comma-separated concurrency sweep (e.g., `1,4,8,16,32`) |

## Metrics

- **Throughput (tok/s)**: Total output tokens / wall clock time.
- **Latency**: End-to-end time per request (avg, p50, p99, max).
- **TTFT**: Time to first token (avg, p50, p99, max).
- **Per-source breakdown**: Metrics broken down by dataset source.

## Dataset Format

Each line in the JSONL file is:
```json
{"prompt": "...", "expected_output_len": 256, "source": "gsm8k"}
```

The `source` field indicates which dataset the prompt came from, enabling per-source analysis and filtering.
