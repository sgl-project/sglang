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

## Profile with PyTorch Profiler
Pytorch Profiler is a convenient basic tool to inspect kernel execution time, call stack, and kernel overlap and occupancy.
- To profile a server
```bash
# set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# start server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# send profiling request from client
python -m sglang.bench_serving --backend sglang --model-path meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile
```
Please make sure that the `SGLANG_TORCH_PROFILER_DIR` should be set at both server and client side, otherwise the trace file cannot be generated correctly . A secure way will be setting `SGLANG_TORCH_PROFILER_DIR` in the `.*rc` file of shell (e.g. `~/.bashrc` for bash shells).

- To profile offline
```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

- View Traces

Trace files can be loaded and visualized from:
1. https://ui.perfetto.dev/ (any browser)
2. chrome://tracing (Chrome browser only)

If browser cannot open trace file due to its large size,
client can generate a small trace file (<100MB) by controlling number of prompts and lengths of prompt outputs.
For example, when profiling a server,
```bash
python -m sglang.bench_serving --backend sglang --model-path meta-llama/Llama-3.1-8B-Instruct --num-prompts 2 --sharegpt-output-len 100 --profile
```
sets the number of prompts to 2 with `--num-prompts` argument and limits the length of output sequences to 100 with `--sharegpt-output-len` argument, which can generate a small trace file for browser to open smoothly.

## Profile with Nsight
Nsight systems is an advanced tool that exposes more profiling details, such as register and shared memory usage, annotated code regions and low-level CUDA APIs and events.

0. Prerequisite: install using apt, or run inside a [NVIDIA Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) or [SGLang Docker container](https://github.com/sgl-project/sglang/tree/main/docker).

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
# launch the server, set the delay and duration times according to needs
# after the duration time has been used up, server will be killed by nsys

nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70 python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache

# client
python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --dataset-name random --random-input 1024 --random-output 512
```

In practice, we recommend users to set `--duration` argument to a large value. Whenever user wants the server to stop profiling. Firstly run:

```bash
nsys sessions list
```

to get the session id in the form of `profile-XXXXX`, then run:

```bash
nsys stop --session=profile-XXXXX
```

to manually kill the profiler and generate `nsys-rep` files instantly.

3. Use NVTX to annotate code regions, e.g. to see their execution time.

```bash
# install nvtx
pip install nvtx
```
``` python
# code snippets
import nvtx
with nvtx.annotate("description", color="color"):
    # some critical code
```

## Other tips
1. You can benchmark a model using dummy weights by only providing the config.json file. This allows for quick testing of model variants without training. To do so, add `--load-format dummy` to the above commands and then you only need a correct `config.json` under the checkpoint folder.
2. You can benchmark a model with modified configs (e.g., less layers) by using `--json-model-override-args`. For example, you can benchmark a model with only 2 layers and 2 kv heads using `python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch 32 --input-len 256 --output-len 32 --load-format dummy --json-model-override-args '{"num_hidden_layers": 1, "num_key_value_heads": 1}'`
3. You can use `--python-backtrace=cuda` to see python call stack for all CUDA kernels, as in PyTorch Profiler. (Caveat: this can cause inaccurately long kernel runtimes for CUDA event based timing)
4. For more args please see https://docs.nvidia.com/nsight-systems/UserGuide/index.html
