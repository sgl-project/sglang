# Video-MME Benchmark

This directory contains scripts to benchmark VLMs on the lmms-lab/Video-MME benchmark with any openai-compatible server.

## Setup

1. **Install Dependencies:**
   You'll need `lmms-eval`
   ```shell
   pip install git+https://github.com/mickqian/lmms-eval.git@4587df0dadb4bc349b93e41add967886dfa87361
   ```

## Usage

1. **Launch an openai-compatible server, using SGLang as an example**
   ```shell
   python -m sglang.launch_server --model-path <your-vlm-model-path> --port 30000
   ```
    
For Video benchmark, we recommend you leave as much memory as possible to avoid OOM

2. **Run the benchmark script:**
   ```shell
   python bench_openai_server --port 30000
   ```
