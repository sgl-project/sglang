# ASR Benchmark

This benchmark evaluates the performance and accuracy (Word Error Rate - WER) of Automatic Speech Recognition (ASR) models served via SGLang.

## Supported Models

- `openai/whisper-large-v3`
- `openai/whisper-large-v3-turbo`

## Setup

Install the required dependencies:

```bash
apt install ffmpeg
pip install librosa soundfile datasets evaluate jiwer transformers openai torchcodec torch
```

## Running the Benchmark

### 1. Start SGLang Server

Launch the SGLang server with a Whisper model:

```bash
python -m sglang.launch_server --model-path openai/whisper-large-v3 --port 30000
```

### 2. Run the Benchmark Script

Basic usage (using chat completions API):

```bash
python bench_sglang.py --base-url http://localhost:30000 --model openai/whisper-large-v3 --n-examples 10
```

Using the OpenAI-compatible transcription API:

```bash
python bench_sglang.py \
    --base-url http://localhost:30000 \
    --model openai/whisper-large-v3 \
    --api-type transcription \
    --language English \
    --n-examples 10
```

Run with streaming and show real-time output:

```bash
python bench_sglang.py \
    --base-url http://localhost:30000 \
    --model openai/whisper-large-v3 \
    --api-type transcription \
    --stream \
    --show-predictions \
    --concurrency 1
```

Run with higher concurrency and save results:

```bash
python bench_sglang.py \
    --base-url http://localhost:30000 \
    --model openai/whisper-large-v3 \
    --concurrency 8 \
    --n-examples 100 \
    --output results.json \
    --show-predictions
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--base-url` | SGLang server URL | `http://localhost:30000` |
| `--model` | Model name on the server | `openai/whisper-large-v3` |
| `--dataset` | HuggingFace dataset for evaluation | `D4nt3/esb-datasets-earnings22-validation-tiny-filtered` |
| `--split` | Dataset split to use | `validation` |
| `--concurrency` | Number of concurrent requests | `4` |
| `--n-examples` | Number of examples to process (`-1` for all) | `-1` |
| `--output` | Path to save results as JSON | `None` |
| `--show-predictions` | Display sample predictions | `False` |
| `--print-n` | Number of samples to display | `5` |
| `--api-type` | API to use: `chat` (chat completions) or `transcription` (audio transcriptions) | `chat` |
| `--language` | Language for transcription API (e.g., `English`, `en`) | `None` |
| `--stream` | Enable streaming mode for transcription API | `False` |

## Metrics

The benchmark outputs:

| Metric | Description |
|--------|-------------|
| **Total Requests** | Number of successful ASR requests processed |
| **WER** | Word Error Rate (lower is better), computed using the `evaluate` library |
| **Average Latency** | Mean time per request (seconds) |
| **Median Latency** | 50th percentile latency (seconds) |
| **95th Latency** | 95th percentile latency (seconds) |
| **Throughput** | Requests processed per second |
| **Token Throughput** | Output tokens per second |

## Example Output

```bash
python bench_sglang.py --api-type transcription --concurrency 128 --model openai/whisper-large-v3 --show-predictions

Loading dataset: D4nt3/esb-datasets-earnings22-validation-tiny-filtered...
Using API type: transcription
Repo card metadata block was not found. Setting CardData to empty.
WARNING:huggingface_hub.repocard:Repo card metadata block was not found. Setting CardData to empty.
Performing warmup...
Processing 511 samples...
------------------------------
Results for openai/whisper-large-v3:
Total Requests: 511
WER: 12.7690
Average Latency: 1.3602s
Median Latency: 1.2090s
95th Latency: 2.9986s
Throughput: 19.02 req/s
Token Throughput: 354.19 tok/s
Total Test Time: 26.8726s
------------------------------

==================== Sample Predictions ====================
Sample 1:
  REF: on the use of taxonomy i you know i think it is it is early days for us to to make any clear indications to the market about the proportion that would fall under that requirement
  PRED: on the eu taxonomy i think it is early days for us to make any clear indications to the market about the proportion that would fall under that requirement
----------------------------------------
Sample 2:
  REF: so within fiscal year 2021 say 120 a 100 depending on what the micro will do and next year it is not necessarily payable in q one is we will look at what the cash flows for 2022 look like
  PRED: so within fiscal year 2021 say $120000 $100000 depending on what the macro will do and next year it is not necessarily payable in q one is we will look at what the cash flows for 2022 look like
----------------------------------------
Sample 3:
  REF: we talked about 4.7 gigawatts
  PRED: we talked about 4.7 gigawatts
----------------------------------------
Sample 4:
  REF: and you know depending on that working capital build we will we will see what that yields
  PRED: and depending on that working capital build we will see what that yields what
----------------------------------------
Sample 5:
  REF: so on on sinopec what we have agreed with sinopec way back then is that free cash flows after paying all capexs are distributed out 30 70%
  PRED: so on sinopec what we have agreed with sinopec way back then is that free cash flows after paying all capexes are distributed out 30% 70%
----------------------------------------
============================================================
```

## Notes

- Audio samples longer than 30 seconds are automatically filtered out (Whisper limitation)
- The benchmark performs a warmup request before measuring performance
- Results are normalized using the model's tokenizer when available
- When using `--stream` with `--show-predictions`, use `--concurrency 1` for clean sequential output
- The `--language` option accepts both full names (e.g., `English`) and ISO 639-1 codes (e.g., `en`)

## Troubleshooting

**Server connection refused**
- Ensure the SGLang server is running and accessible at the specified `--base-url`
- Check that the port is not blocked by a firewall

**Out of memory errors**
- Reduce `--concurrency` to lower GPU memory usage
- Use a smaller Whisper model variant
