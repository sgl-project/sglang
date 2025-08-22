Audio ASR Benchmark (SGLang)

This benchmark evaluates audio transcription-style requests against the SGLang OpenAI-compatible chat endpoint using multimodal audio inputs. It follows the style of existing benchmarks (e.g., MMMU) and uses an async streaming client to measure TTFT, per-chunk inter-token latencies, and end-to-end latency.

Supported models include audio-capable MLLMs such as MiniCPM-o and Gemma3n. The default example below uses a relatively small model: `openbmb/MiniCPM-o-2_6`.

Quick start
- Start server:
  - `python -m sglang.launch_server --model-path openbmb/MiniCPM-o-2_6 --port 30000 --trust-remote-code`
- Run benchmark:
  - `python benchmark/asr/bench_sglang_asr.py --port 30000 --dataset openslr/librispeech_asr --split test --limit 8 --concurrency 4`

Notes
- The benchmark sends audio as a data URL (`data:audio/wav;base64,...`) in `messages[].content[].audio_url.url`, so no local files need to be exposed to the server.
- By default, audio clips longer than 30 seconds are skipped to keep runs fast and consistent.
- You can point to other Hugging Face ASR datasets via `--dataset` and `--split`. Only a small number of samples is used by default; increase `--limit` as needed.

Example for Gemma3n (if weights are available locally)
- Start server:
  - `python -m sglang.launch_server --model-path google/gemma-3-**YOUR_VARIANT** --port 30000 --trust-remote-code`
- Run benchmark:
  - `python benchmark/asr/bench_sglang_asr.py --port 30000 --dataset openslr/librispeech_asr --split test --limit 8 --concurrency 4`

CLI options
- `--port`: SGLang server port (default: 30000)
- `--dataset`: HF dataset path (default: `openslr/librispeech_asr`)
- `--split`: split name (default: `test`)
- `--subset`: dataset config/subset name if required by the dataset (default: empty)
- `--limit`: max number of samples (default: 8)
- `--concurrency`: parallel streaming requests (default: 4)
- `--max-tokens`: max generated tokens (default: 128)
- `--skip-long`: skip >30s audios (default: true)

Output
- Prints aggregate stats and per-request summaries including success/failure, TTFT, total latency, and output token count if provided in the streamed usage event.

