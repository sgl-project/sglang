# Inductor Compilation Speedup Charts

## Usage

```bash
# Plot speedup charts for a given profile directory
# (baseline defaults to inductor[None])
python profiles/plot_speedup.py profiles/openai/gpt-oss-20b-bf16
python profiles/plot_speedup.py profiles/Qwen/Qwen3-30B-A3B

# Custom baseline
python profiles/plot_speedup.py profiles/Qwen/Qwen3-30B-A3B --baseline "inductor[rmsnorm]"

# Custom model name and output path
python profiles/plot_speedup.py profiles/openai/gpt-oss-20b-bf16 \
    --model-name "lmsys/gpt-oss-20b-bf16" \
    -o my_chart.png
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `directory` | *(required)* | Directory containing `inductor[*].jsonl` profile files |
| `--baseline` | `inductor[None]` | Baseline config name to compute speedup against |
| `--model-name` | derived from directory path | Model name shown in the chart title |
| `-o`, `--output` | `<directory>/speedup_charts.png` | Output PNG path |

## Profile format

Each JSONL file contains one JSON object per line with fields from `bench_one_batch.py`:

```json
{"run_name": "default", "batch_size": 1, "input_len": 1024, "output_len": 8192, "prefill_latency": ..., "prefill_throughput": ..., "median_decode_latency": ..., "median_decode_throughput": ..., "total_latency": ..., "overall_throughput": ...}
```

Files are named `b1b-moe[auto]-inductor[<config>].jsonl` where `<config>` identifies the inductor compilation scope (e.g. `None`, `moe`, `rmsnorm`, `moe-rmsnorm`, `topk-moe-rmsnorm`).
