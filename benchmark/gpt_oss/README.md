# How to reproduce the result of GPT-OSS with SGLang

### Install the latest SGLang

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.1.post3

pip install --upgrade pip
pip install -e "python[all]"
```

### Reproduce the benchmark throughput result (Batch Size 1)

Launch Command

```bash
# MXFP4 120B on H100
python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 8 --attention-backend triton

# BF16 120B on H100
python3 -m sglang.launch_server --model lmsys/gpt-oss-120b-bf16 --tp 8 --attention-backend triton

# MXFP4 120B on B200
python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 4

# BF16 120B on B200
python3 -m sglang.launch_server --model lmsys/gpt-oss-120b-bf16 --tp 4
```

Benchmark Command

```bash

# MXFP4 120B on H100
python3 -m sglang.bench_one_batch_server --model openai/gpt-oss-120b --base-url http://localhost:30000 --batch-size 1 --input-len 1024 --output-len 512 --show-report
```

### Reproduce the benchmark throughput result (Batch Size 32)

Launch Command

```bash
# MXFP4 120B on H100
python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 8

# BF16 120B on H100
python3 -m sglang.launch_server --model lmsys/gpt-oss-120b-bf16 --tp 8

# MXFP4 120B on B200
python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 4

# BF16 120B on B200
python3 -m sglang.launch_server --model lmsys/gpt-oss-120b-bf16 --tp 4
```

Benchmark Command

```bash
python3 -m sglang.bench_one_batch_server --model openai/gpt-oss-120b --base-url http://localhost:30000 --batch-size 32 --input-len 1024 8192 --output-len 512 --show-report
```

### Reproduce the evaluation result

Install gpt-oss

```bash
git clone https://github.com/openai/gpt-oss.git
cd gpt-oss
pip install -e .
```

Evaluation Command

```bash
DATASET=gpqa
BASE_URL=YOUR_BASE_URL
OPENAI_API_KEY=dummy python -m gpt_oss.evals \
    --base-url ${BASE_URL}/v1 \
    --model dummy \
    --reasoning-effort low,medium,high \
    --eval $DATASET \
    --n-threads 1000
```

### Reproduce the benchmark result of acceptance length
> Note: On B200, if top k is 1, set `--attention-backend trtllm_mha`
```bash
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge/benchmarks
config_list=(
    "1,0,0,0"
    "1,3,1,4"
    "1,5,4,8"
)
python3 bench_model_speedup.py \
    --model-path openai/gpt-oss-120b \
    --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --attention-backend fa3 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 \
    --output lmsys_gpt-oss-120b_Eagle3_result.jsonl

python3 bench_model_speedup.py \
    --model-path openai/gpt-oss-120b \
    --speculative-draft-model-path nvidia/gpt-oss-120b-Eagle3 \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --attention-backend fa3 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 \
    --output nv_gpt-oss-120b_Eagle3_result.jsonl
```

### Reproduce the result of speculative decoding speedup

Launch Command

```bash
# On Hopper:
# - Tree decoding (topk > 1) and chain decoding (topk = 1) are supported on both FA3 and Triton backends.
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algorithm EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --tp 4
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algorithm EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 --tp 4

# On Blackwell:
# - Chain decoding (topk = 1) is supported on TRTLLM-MHA backend. Tree decoding (topk > 1) is in progress, stay tuned!
# - Both tree decoding (topk > 1) and chain decoding (topk = 1) are supported on the Triton backend.
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algo EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --tp 4
python3 -m sglang.launch_server --model openai/gpt-oss-120b --speculative-algo EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 --speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 --attention-backend triton --tp 4
```

Benchmark Command

```bash
config_list=(
    "1,0,0,0"
    "1,3,1,4"
    "1,5,4,8"
)
python3 bench_model_speedup.py \
    --model-path openai/gpt-oss-120b \
    --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --attention-backend fa3 \
    --config-list "${config_list[@]}" \
    --benchmark-list gsm8k:200 humaneval:200 math500:200 \
    --output lmsys_gpt-oss-120b_Eagle3_result.jsonl
```

We can gain the best speedup with the following settings:

- **1.39x** speedup with the `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` setting.
- **1.52x** speedup with the `--speculative-num-steps 5 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8` setting.
