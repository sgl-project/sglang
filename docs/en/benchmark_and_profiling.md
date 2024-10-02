# Benchmark and Profiling

## Benchmark
- Benchmark a single static batch by running the following command without launching a server. The arguments are the same as for `launch_server.py`. Note that this is not a dynamic batching server, so it may run out of memory for a batch size that a real server can handle. A real server truncates the prefill into several batches, while this unit test does not. For accurate large batch testing, consider using `sglang.bench_serving`.
  ```
  python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 32 --input-len 256 --output-len 32
  ```
- Benchmark online serving. Launch a server first and run the following command.
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

1. To profile a single batch, use `nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_latency --model meta-llama/Meta-Llama-3-8B --batch-size 64 --input-len 512`

2. To profile a server, e.g.

```bash
# server
# set the delay and duration times according to needs
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70 python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache

# client
python3 -m sglang.bench_serving --backend sglang --num-prompts 6000 --dataset-name random --random-input 4096 --random-output 2048
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