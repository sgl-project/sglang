# Evaluating New Models with SGLang

This document provides a comprehensive guide for evaluating new models using SGLang before open-sourcing or deployment. It covers installation, server launching, and evaluating both accuracy and performance (latency and throughput) for Large Language Models and Multimodal Language Models.

## Installation

To get started, clone the SGLang repository and install the dependencies.

```bash
git clone -b <your-branch> https://github.com/<your-fork>/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python"
```

## Launching the Server

Before running evaluations, you need to launch the SGLang server. The command below shows a general usage pattern. You should adjust flags based on your specific model and hardware requirements.

```bash
python -m sglang.launch_server \
  --model-path <model-path> \
  --host 0.0.0.0 \
  --port 30000 \
```

**Common Flags:**
- `--model-path`: Path to the model weights or Hugging Face model ID.
- `--tp-size`: Tensor parallelism size (number of GPUs).
- `--ep-size`: Expert parallelism size (for MoE models).
- `--tool-call-parser`: Specify parser for tool use (e.g., `minimax-m2`).
- `--reasoning-parser`: Specify parser for reasoning models (e.g., `minimax-append-think`).
- `--trust-remote-code`: Required for some custom models.

> [!NOTE]
> Record the exact GPU type, count, and any specific flags used for reproducibility.

## Accuracy Evaluation

### Large Language Models (LLMs)

SGLang provides built-in scripts to evaluate common benchmarks.

#### MMLU
[Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300).

```bash
python -m sglang.test.run_eval \
  --eval-name mmlu \
  --port 30000 \
  --num-examples 1000
```

#### GSM8K
[Grade School Math 8K](https://arxiv.org/abs/2110.14168).

```bash
python -m sglang.test.few_shot_gsm8k \
  --host http://127.0.0.1 \
  --port 30000 \
  --num-questions 200 \
  --num-shots 5
```

#### HellaSwag
[HellaSwag](https://arxiv.org/abs/1905.07830) Common sense reasoning.

```bash
python3 benchmark/hellaswag/bench_sglang.py \
  --host http://127.0.0.1 \
  --port 30000 \
  --num-questions 200 \
  --num-shots 20
```

#### GPQA
[Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022).

```bash
python3 -m sglang.test.run_eval \
  --eval-name gpqa \
  --port 30000 \
  --num-examples 198 \
  --max-tokens 120000 \
  --repeat 8
```
> [!TIP]
> For reasoning models, add `--thinking-mode <mode>` (e.g., `qwen3`, `deepseek-r1`, `deepseek-v3`). You may skip it if the model has forced thinking enabled.

#### HumanEval
HumanEval problem solving dataset described in the paper "[Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)".

```bash
# Install human-eval if it is not installed yet 
pip install human_eval 

python3 -m sglang.test.run_eval \
  --eval-name humaneval \
  --num-examples 10 \
  --port 30000
```

### Multimodal Language Models

#### MMMU
[Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark](https://arxiv.org/abs/2311.16502).

```bash
python3 benchmark/mmmu/bench_sglang.py \
  --port 30000 \
  --concurrency 64
```
> [!TIP]
> You can set max tokens by passing `--extra-request-body '{"max_tokens": 4096}'`.

We also encourage evaluating on `Video-MME`, `DocVQA`, and other relevant benchmarks if your model supports them.

## Performance Evaluation

Performance benchmarks measure **Latency** (Time To First Token - TTFT) and **Throughput** (tokens/second).

**For VLMs:**
The following sections are examples for LLMs. For VLMs, add the following arguments to the commands:
```bash
--backend sglang-oai-chat
--dataset-name image
--image-count 2
--image-resolution 720p
--random-input-len 128
```

### Latency-Sensitive Benchmark

This simulates a scenario with low concurrency (e.g., single user) to measure latency.

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --dataset-name random \
  --num-prompts 10 \
  --max-concurrency 1
```

### Throughput-Sensitive Benchmark

This simulates a high-traffic scenario to measure maximum system throughput.

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --dataset-name random \
  --num-prompts 1000 \
  --max-concurrency 100
```

### Single Batch Performance
You can also benchmark the performance of processing a single batch offline.

```bash
python -m sglang.bench_one_batch_server \
  --model <model-path> \
  --batch-size 8 \
  --input-len 1024 \
  --output-len 1024
```

You can run more granular benchmarks:

- **Low Concurrency**: `--num-prompts 10 --max-concurrency 1`
- **Medium Concurrency**: `--num-prompts 80 --max-concurrency 16`
- **High Concurrency**: `--num-prompts 500 --max-concurrency 100`

## Reporting Results

For each evaluation, please report:
1.  **Metric Score**: Accuracy %, Latency (ms), Throughput (tok/s).
2.  **Environment settings**: GPU type/count, SGLang commit hash.
3.  **Launch configuration**: Model path, TP size, and any special flags.
4.  **Evaluation parameters**: Number of shots, examples, max tokens.
