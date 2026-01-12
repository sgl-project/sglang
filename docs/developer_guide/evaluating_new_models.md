# Evaluating New Models with SGLang

This document provides a reusable evaluation template for new models before open-sourcing. It is organized as a matrix over model type (LLM/VLM) and goal (Accuracy/Performance).

## Install

```bash
git clone -b <your-branch> https://github.com/<your-fork>/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python"
```

## Launch the server

Use model-specific flags (tool-call parser, reasoning parser, TP/EP sizes, etc.). Example for MiniMax M2:

```bash
python -m sglang.launch_server \
  --model-path MiniMaxAI/MiniMax-M2 \
  --tp-size 8 \
  --ep-size 8 \
  --tool-call-parser minimax-m2 \
  --trust-remote-code \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.85
```

Notes:
- Record the exact GPU type, count, and any unsupported flags.

## Accuracy benchmarks

### For LLM

#### MMLU:
```bash
python -m sglang.test.run_eval --eval-name mmlu --port 30000 --num-examples 1000
```

#### GSM8K:
```bash
python -m sglang.test.few_shot_gsm8k --host http://127.0.0.1 --port 30000 --num-questions 200 --num-shots 5
```

#### HellaSwag:
```bash
cd benchmark/hellaswag
python3 bench_sglang.py --num-questions 200 --num-shots 20 --host http://127.0.0.1 --port 30000
```

#### GPQA:

```bash
python3 -m sglang.test.run_eval  \
  --port 30000 \
  --eval-name gpqa \
  --thinking-mode qwen3 \  # Select for thinking model, supported thinking mode: "deepseek-r1", "deepseek-v3", "qwen3"
  --num-examples 198 \
  --max-tokens 120000 \
  --repeat 8
```

### For VLM

#### MMMU
```bash
cd benchmark/mmmu
python3 bench_sglang.py --port 30000 --concurrency 16
```

### Record and report results

For each benchmark, report:
- Score and evaluation settings (shots, num examples, max tokens, repeat count)
- Total latency when applicable (e.g., GPQA)
- Hardware (GPU type/count, driver, CUDA), model revision, and SGLang commit hash
- Any model-specific flags used to launch the server

## Performance

### Online throughput
Launch a server first:
```bash
python3 -m sglang.bench_serving --backend sglang --num-prompt 100
```

Static batch latency via server:
```bash
python -m sglang.bench_one_batch_server \
  --model finetuned_model \
  --batch-size 8 \
  --input-len 1024 \
  --output-len 1024
```

For more detailed benchmark and profiling methodology, refer to:
- [Bench Serving Guide](./bench_serving.md)
- [Benchmark and Profiling](./benchmark_and_profiling.md)
