# Benchmark and Profiling

## Benchmark

- Benchmark the latency of running a single static batch without a server. The arguments are the same as for `launch_server.py`.
  Note that this is a simplified test script without a dynamic batching server, so it may run out of memory for a batch size that a real server can handle. A real server truncates the prefill into several batches, while this simplified script does not.
  - Without a server (do not need to launch a server)
    ```bash
    python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch 32 --input-len 256 --output-len 32
    ```
  - With a server (please use `sglang.launch_server` to launch a server first and run the following command.)
    ```bash
    python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch-size 32 --input-len 256 --output-len 32
    ```


- Benchmark offline processing. This script will start an offline engine and run the benchmark.

  ```bash
  python3 -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --num-prompts 10
  ```

- Benchmark online serving. Please use `sglang.launch_server` to launch a server first and run the following command.

  ```bash
  python3 -m sglang.bench_serving --backend sglang --num-prompt 10
  ```

## Profile with PyTorch Profiler

[Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) is a convenient basic tool to inspect kernel execution time, call stack, and kernel overlap and occupancy.

### Profile a server with `sglang.bench_serving`

```bash
# set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# start server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# send profiling request from client
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile
```

Please make sure that the `SGLANG_TORCH_PROFILER_DIR` should be set at both server and client side, otherwise the trace file cannot be generated correctly . A secure way will be setting `SGLANG_TORCH_PROFILER_DIR` in the `.*rc` file of shell (e.g. `~/.bashrc` for bash shells).

For more details, please refer to [Bench Serving Guide](./bench_serving.md).

### Profile In PD Disaggregation Mode

When profiling in PD disaggregation mode, prefill and decode workers **must be profiled separately** due to torch profiler limitations. The `bench_serving` command provides dedicated options for this:

#### Profile Prefill Workers

```bash
# set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# start prefill and decode servers (see PD disaggregation docs for setup)
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode prefill
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode decode --port 30001 --base-gpu-id 1

# start router
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000

# send profiling request targeting prefill workers
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000
```

#### Profile Decode Workers

```bash
# send profiling request targeting decode workers
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001
```

#### Important Notes

- `--profile-prefill-url` and `--profile-decode-url` are **mutually exclusive** - you cannot profile both at the same time
- Both options support multiple worker URLs for multi-instance setups:
  ```bash
  # Profile multiple prefill workers
  python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000 http://127.0.0.1:30002

  # Profile multiple decode workers
  python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001 http://127.0.0.1:30003
  ```
- Make sure `SGLANG_TORCH_PROFILER_DIR` is set on all worker nodes before starting the servers
- For more details on setting up PD disaggregation, see [PD Disaggregation Guide](../advanced_features/pd_disaggregation.md)

### Profile a server with `sglang.bench_offline_throughput`
```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# profile one batch with bench_one_batch.py
# batch size can be controlled with --batch argument
python3 -m sglang.bench_one_batch --model-path meta-llama/Llama-3.1-8B-Instruct --batch 32 --input-len 1024 --output-len 10 --profile

# profile multiple batches with bench_offline_throughput.py
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

### Profile a server with `sglang.profiler`

When the server is running (e.g., processing a decoding request), you can start live profiling immediately by sending a profile request to the server.

You can do this by running `python3 -m sglang.profiler`. For example:

```
# Terminal 1: Send a generation request
python3 -m sglang.test.send_one

# Terminal 2: Before the above request finishes, quickly launch the following command in a separate terminal.
# It will generate a profile of the above request for several decoding batches.
python3 -m sglang.profiler
```

You can also combine the above operations into a single command

```
python3 -m sglang.test.send_one --profile
```

### Profiler Trace Merger for Distributed Traces

SGLang now supports automatic merging of profiling traces from distributed setups with multiple parallelism types (TP, DP, PP, EP). This feature is particularly useful for analyzing performance across distributed runs.

#### Multi-Node Profiling and Shared Storage Considerations

Single-node profiler output merging is completely supported. When profiling in distributed environments spanning multiple nodes, shared storage (e.g., NFS, Lustre) should be accessible by all nodes for the output directory to enable merging of trace files.

If there is no shared storage accessible across nodes, automatic merging of trace files during profiling is not supported directly as of now.

#### HTTP API Usage

```bash
# Start profiling with automatic trace merging enabled
curl -X POST <BASE_URL>/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles", # where to store profile traces
    "num_steps": 10,
    "activities": ["CPU", "GPU"],
    "merge_profiles": true # optional argument to merge profile traces (default=False)
  }'
```

#### Command Line Usage

```bash
# Start profiling with merge enabled
python -m sglang.profiler \
  --num-steps 10 \
  --cpu \
  --gpu \
  --output-dir /tmp/profiles \
  --merge-profiles # optional argument to merge profile traces (default=False)
```

#### Output Files

The profile merger generates:
- Individual rank trace files: `{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`
- Merged trace file: `merged-{profile_id}.trace.json.gz`

### Possible PyTorch bugs
If in any cases you encounter the following error (for example, using qwen 2.5 VL):
```bash
RuntimeError: !stack.empty() INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/autograd/profiler_python.cpp":983, please report a bug to PyTorch. Python replay stack is empty.
```
This is likely a PyTorch Bug reported in [Bug: vLLM Profiler](https://github.com/vllm-project/vllm/issues/18240) and [Bug: torch.profiler.profile](https://github.com/pytorch/pytorch/issues/101632). As a workaround, you may disable `with_stack` with an environment variable such as follows:
```bash
export SGLANG_PROFILE_WITH_STACK=False
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

### View traces

Trace files can be loaded and visualized from:

1. https://ui.perfetto.dev/ (any browser)
2. chrome://tracing (Chrome browser only)

If browser cannot open trace file due to its large size,
client can generate a small trace file (<100MB) by controlling number of prompts and lengths of prompt outputs.
For example, when profiling a server,

```bash
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 2 --sharegpt-output-len 100 --profile
```

This command sets the number of prompts to 2 with `--num-prompts` argument and limits the length of output sequences to 100 with `--sharegpt-output-len` argument, which can generate a small trace file for browser to open smoothly.

Additionally, if you want to locate the SGLang Python source code through the cuda kernel in Trace, you need to disable CUDA Graph when starting the service. This can be done by using the `--disable-cuda-graph` parameter in the command to start the service.

## Profile with Nsight

[Nsight systems](https://docs.nvidia.com/nsight-systems/) is an advanced tool that exposes more profiling details, such as register and shared memory usage, annotated code regions and low-level CUDA APIs and events.

1. Prerequisite:

   Install using apt, or run inside a [NVIDIA Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) or [SGLang Docker container](https://github.com/sgl-project/sglang/tree/main/docker).

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

2. To profile a single batch, use

   ```bash
   nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model meta-llama/Meta-Llama-3-8B --batch-size 64 --input-len 512
   ```

3. To profile a server, e.g.

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

4. Use NVTX to annotate code regions, e.g. to see their execution time.

   ```bash
   # install nvtx
   pip install nvtx
   ```

   ```python
   # code snippets
   import nvtx
   with nvtx.annotate("description", color="color"):
       # some critical code
   ```

## Other tips

1. You can benchmark a model using dummy weights by only providing the config.json file. This allows for quick testing of model variants without training. To do so, add `--load-format dummy` to the above commands and then you only need a correct `config.json` under the checkpoint folder.
2. You can benchmark a model with modified configs (e.g., less layers) by using `--json-model-override-args`. For example, you can benchmark a model with only 2 layers and 2 kv heads using:

   ```bash
   python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch 32 --input-len 256 --output-len 32 --load-format dummy --json-model-override-args '{"num_hidden_layers": 1, "num_key_value_heads": 1}'
   ```

3. You can use `--python-backtrace=cuda` to see python call stack for all CUDA kernels, as in PyTorch Profiler. (Caveat: this can cause inaccurately long kernel runtimes for CUDA event based timing)
4. For more arguments see [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
