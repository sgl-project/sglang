# Benchmark and Profiling

## Benchmark
- Benchmark the latency of running a single static batch without a server. The arguments are the same as for `launch_server.py`.
  Note that this is a simplified test script without a dynamic batching server, so it may run out of memory for a batch size that a real server can handle. A real server truncates the prefill into several batches, while this simplified script does not.
  ```
  python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch 32 --input-len 256 --output-len 32
  ```
- Benchmark offline processing. This script will start an offline engine and run the benchmark.
  ```
  python3 -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --num-prompts 10
  ```
- Benchmark online serving. Please use `sglang.launch_server` to launch a server first and run the following command.
  ```
  python3 -m sglang.bench_serving --backend sglang --num-prompt 10
  ```

## Profile with Nsight
0. Prerequisite
```bash
# install nsys
# https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

1. To profile a single batch, use `nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model meta-llama/Meta-Llama-3-8B --batch-size 64 --input-len 512`

2. To profile a server, e.g.

```bash
# server
# set the delay and duration times according to needs
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70 python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache

# client
python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --dataset-name random --random-input 1024 --random-output 512
```

3. Use NVTX, e.g.

```bash
# install nvtx
pip install nvtx

# code snippets
import nvtx
with nvtx.annotate("description", color="color"):
    # some critical code
```

## Other tips

1. You can benchmark a model using dummy weights by only providing the config.json file. This allows for quick testing of model variants without training. To do so, add `--load-format dummy` to the above commands and then you only need a correct `config.json` under the checkpoint folder.

## Profile with PyTorch Profiler
- To profile a server
```bash
# set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
# start server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

python -m sglang.bench_serving --backend sglang --model-path meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --profile
```

Traces can be visualized using https://ui.perfetto.dev/.

- To profile offline
```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```
