## Download Dataset

```sh
wget -O question.jsonl \
  https://raw.githubusercontent.com/lm-sys/FastChat/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl
```

## Run benchmark

### Benchmark sglang

```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_sglang.py --num-questions 80
```

### Benchmark sglang EAGLE

```
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct --speculative-algo EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 5 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --dtype float16 --port 30000
```

```
python3 bench_sglang_eagle.py --num-questions 80 --parallel 1
```

### Benchmark vllm

```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests --port 21000
```

```
python3 bench_other.py --num-questions 80 --backend vllm
```

### Benchmark lightllm

```
# A10G
python -m lightllm.server.api_server --tokenizer_mode auto --model_dir ~/model_weights/llama-2-7b-chat-hf --max_total_token_num 16000 --port 22000
```

```
python3 bench_other.py --num-questions 80 --backend lightllm
```

## Reproducible ECHO A/B benchmark

`bench_sglang_eagle.py` runs all 80 MT-Bench conversations (160 generation
turns) through SGLang's OpenAI chat endpoint. The endpoint applies the target
model's Hugging Face chat template, so local model paths do not depend on
name-based template detection.

The script uses a commit-pinned copy of FastChat's question set and validates
that a full run has 80 unique question IDs, two turns per question, and ten
questions in each of the eight categories. It downloads the pinned file when
`--question-file` is omitted. To prepare it manually:

```bash
wget -O /tmp/mt_bench_question.jsonl \
  https://raw.githubusercontent.com/lm-sys/FastChat/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl
```

### Launch the server

The baseline and ECHO servers must use the same arguments. The ECHO server adds
only `--speculative-echo-threshold`. For example:

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/target \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /path/to/draft \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --speculative-num-draft-tokens 32 \
  --attention-backend fa3 \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --dtype bfloat16 \
  --tp-size 1 \
  --cuda-graph-max-bs-decode 32 \
  --port 30000
```

Disable the radix cache for performance A/B runs. Otherwise, a configuration
whose own turn-one answer matches the frozen history can reuse KV cache on turn
two while another configuration must prefill it, biasing the comparison.

### Freeze the two-turn inputs

First run the dense server once to prepare a canonical turn-one history. This
preparation run is not a timed result:

```bash
python3 benchmark/mtbench/bench_sglang_eagle.py \
  --host 127.0.0.1 \
  --port 30000 \
  --question-file /tmp/mt_bench_question.jsonl \
  --num-questions 80 \
  --parallel 32 \
  --max-new-tokens 1024 \
  --num-gpus 1 \
  --run-label canonical-input \
  --answer-file /tmp/eagle3_canonical_answers.jsonl \
  --result-file /tmp/eagle3_preparation.jsonl
```

Pass that same file to every timed baseline and ECHO run:

```bash
python3 benchmark/mtbench/bench_sglang_eagle.py \
  --host 127.0.0.1 \
  --port 30000 \
  --question-file /tmp/mt_bench_question.jsonl \
  --num-questions 80 \
  --parallel 32 \
  --max-new-tokens 1024 \
  --warmup-questions 4 \
  --num-gpus 1 \
  --frozen-turn-one-file /tmp/eagle3_canonical_answers.jsonl \
  --run-label eagle3-dense \
  --answer-file /tmp/eagle3_dense_answers.jsonl \
  --raw-result-file /tmp/eagle3_dense_raw.jsonl \
  --result-file /tmp/eagle3_results.jsonl
```

Restart the server with the same arguments plus:

```bash
--speculative-echo-threshold 0.2
```

Then repeat the timed command with a different run label and output files while
keeping `--frozen-turn-one-file` unchanged. Run each timed configuration three
times and report the median. The result file is append-only.

`--frozen-turn-one-file` gives every configuration the same assistant history
for turn two. This prevents a turn-one output difference from changing the
turn-two input during a performance comparison.

### CUDA Graph correctness

Graph/eager exact parity is a separate correctness gate, not a performance
requirement. Launch the ECHO server with
`--enable-deterministic-inference` and
`--cuda-graph-backend-decode disabled`, then save its answer file using the
same frozen history and request settings:

```bash
python3 benchmark/mtbench/bench_sglang_eagle.py \
  --host 127.0.0.1 \
  --port 30000 \
  --question-file /tmp/mt_bench_question.jsonl \
  --num-questions 80 \
  --parallel 32 \
  --max-new-tokens 1024 \
  --frozen-turn-one-file /tmp/eagle3_canonical_answers.jsonl \
  --run-label eagle3-eager-correctness \
  --answer-file /tmp/eagle3_eager_answers.jsonl \
  --result-file /tmp/eagle3_correctness.jsonl
```

Restart with decode CUDA Graph enabled and run:

```bash
python3 benchmark/mtbench/bench_sglang_eagle.py \
  --host 127.0.0.1 \
  --port 30000 \
  --question-file /tmp/mt_bench_question.jsonl \
  --num-questions 80 \
  --parallel 32 \
  --max-new-tokens 1024 \
  --frozen-turn-one-file /tmp/eagle3_canonical_answers.jsonl \
  --reference-answer-file /tmp/eagle3_eager_answers.jsonl \
  --require-exact-match \
  --run-label eagle3-graph-correctness \
  --answer-file /tmp/eagle3_graph_answers.jsonl \
  --result-file /tmp/eagle3_correctness.jsonl
```

Do not use deterministic inference for published performance numbers.
Target-only and speculative decoding can legally diverge after small BF16
rounding differences from different GEMM shapes, so their exact-text rate is a
diagnostic rather than the graph correctness gate.

### Sampling policy and metrics

By default, the serving benchmark uses temperature 0, seed 0, normal EOS
handling, and `enable_thinking=False`. This is a greedy, controlled MT-Bench
workload for parity and performance comparisons; it is not an MT-Bench quality
score. Greedy sampling alone does not guarantee token-identical output across
different BF16 batch or GEMM shapes.

Use `--temperature-policy official` for FastChat's category-specific
temperatures (0.7 for writing and roleplay, 0.1 for STEM and humanities, and 0
for the other categories). Add `--enable-thinking` to benchmark a model's
thinking mode. Record these choices when publishing results.

The result JSONL contains overall, per-turn, and per-category values for:

- prompt, cached, and completion tokens;
- output throughput and conversations per second;
- request and complete-conversation latency percentiles;
- token-weighted speculative acceptance length and acceptance rate;
- ECHO runtime metadata when exposed by the backend;
- request errors and length-truncated responses;
- exact output parity against an optional reference.

The answer JSONL follows FastChat's MT-Bench answer schema. The optional raw
JSONL writes one record per conversation, with per-turn metadata and output
token IDs for deeper diagnosis.
