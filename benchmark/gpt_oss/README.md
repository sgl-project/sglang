# How to reproduce the result of GPT-OSS with SGLang

## Prerequisite

### Install the latest SGLang

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.1.post3

pip install --upgrade pip
pip install -e "python[all]"
```

### Install gpt-oss

```bash
git clone https://github.com/openai/gpt-oss.git
cd gpt-oss
pip install -e .
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
