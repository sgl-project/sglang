# RISC-V Vector (RVV)

This document describes how to set up the [SGLang](https://github.com/sgl-project/sglang) environment
and run LLM inference on RISC-V processors with the RVV (RISC-V Vector Extension) backend.

SGLang's RVV backend targets RISC-V processors implementing **RVV v1.0** with **VLEN ≥ 128**.
It has been verified on the **[SpacemiT K1](https://www.spacemit.com/spacemit-k1/)**
(VLEN=256, as found on the [Banana Pi BPI-F3](https://wiki.banana-pi.org/Banana_Pi_BPI-F3)).
The backend uses custom hand-written RVV intrinsic kernels compiled with **Clang 19+**.
VLEN is auto-detected from `/proc/cpuinfo` at build time (128 / 256 / 512 / 1024 supported).

> **Hardware requirement:** A RISC-V board with RVV v1.0 support and VLEN ≥ 128.
> The instructions below assume a Debian/Ubuntu-based RISC-V Linux system.

## Supported Models

The following models have been verified on the SpacemiT K1 (VLEN=256):

| Model Name | BF16 | W8A8 INT8 |
|:---:|:---:|:---:|
| Llama-3.2-1B | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | - |
| Qwen2.5-1.5B | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | [RedHatAI/Qwen2.5-1.5B-quantized.w8a8](https://huggingface.co/RedHatAI/Qwen2.5-1.5B-quantized.w8a8) |

## Installation

Choose the path that matches your use case:

| I want to… | Follow |
|---|---|
| Run inference without touching kernel code | [Setup](#setup) |
| Modify RVV kernels and recompile | [Kernel Developers](#kernel-developers) |

---

### Setup

The image contains native riscv64 binaries (PyTorch, sgl-kernel) and must be built
**on the RISC-V board itself** — cross-compilation is not supported.
Expect the build to take **30–120 minutes** depending on board speed and network.

**Step 1 — Clone the repository and build the image**

```bash
# On the RISC-V board
git clone https://github.com/sgl-project/sglang.git
cd sglang

> If you are using Podman, you can set an alias so all commands below work unchanged:
> ```bash
> alias docker=podman
> ```


docker build \
    --format docker \
    -t sglang-rvv:latest \
    -f docker/rvv.Dockerfile .

```


> Pass `--format docker` when using Podman to suppress OCI compatibility warnings
> about the `SHELL` instruction.

Run a quick smoke test before launching the server:

```bash
docker run --rm sglang-rvv:latest python3 -c "import sgl_kernel; print('sgl_kernel import OK')"
```

If this command fails, the image build finished but the native `sgl_kernel` binary is
not loadable. Rebuild after syncing to the latest branch tip and ensure the RVV
`sgl-kernel` build step completes without warnings/errors.

---

### Kernel Developers

This workflow lets you edit RVV kernels on the host, recompile inside the container,
and test immediately — without rebuilding the image each time.

**How it works:** The container has the pre-built SGLang packages installed in `/opt/.venv`.
When you mount your local repository, you must either point Python at the mounted source
or install it in editable mode. Otherwise, Python will keep importing the pre-built package.
C++ kernel changes still require recompilation.

**Step 1 — Build the image**

Follow [Setup](#setup) above to build `sglang-rvv:latest` on the RISC-V board.

**Step 2 — Start a persistent dev container**

```bash
mkdir -p ~/.cache/huggingface

docker run -d \
    --name sglang-rvv-dev \
    --network host \
    -v $(pwd):/workspace/sglang \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e "HF_TOKEN=<your_token>" \
    sglang-rvv:latest \
    sleep infinity
```

The container runs in the background with `sleep infinity` so it stays alive across
compile/test cycles. Your local repository is mounted at `/workspace/sglang` inside
the container.

**Step 3 — Enter the container**

```bash
docker exec -it sglang-rvv-dev bash
```

The container stays alive when you exit. To stop it: `docker stop sglang-rvv-dev`.
To resume after a reboot: `docker start sglang-rvv-dev`.

**Step 4 — Make Python code live (recommended)**

Pick one of the following options so Python uses the mounted repo:

**Option A: Use `PYTHONPATH` (fast, no install)**

```bash
export PYTHONPATH=/workspace/sglang/python
```

**Option B: Editable install (recommended for daily dev)**

```bash
cd /workspace/sglang/python
cp pyproject_cpu.toml pyproject.toml
uv pip install -e . --no-deps --index-strategy unsafe-best-match
```

If you do not use `PYTHONPATH` or an editable install, you must reinstall the Python
package after any Python changes.

**Step 5 — Recompile `sgl-kernel` (inside the container)**

Run this whenever you modify C++ kernel code under `sgl-kernel/csrc/`.
It will take about 8 mins on Banana Pi K1.
VLEN is auto-detected from `/proc/cpuinfo` during the CMake step.
For RVV kernel builds, use `pyproject_riscv64.toml` (not `pyproject_cpu.toml`).

```bash
source /opt/.venv/bin/activate
cd /workspace/sglang/sgl-kernel

uv pip uninstall sglang-kernel-riscv64 || true
uv pip uninstall sglang-kernel-cpu || true
uv pip uninstall sgl-kernel-cpu || true
rm -rf build/ _skbuild/ *.egg-info CMakeCache.txt CMakeFiles/
cp pyproject_riscv64.toml pyproject.toml

if command -v clang-19 >/dev/null 2>&1; then
    export CC=clang-19
    export CXX=clang++-19
else
    export CC=clang
    export CXX=clang++
fi
uv pip install . --no-deps --no-build-isolation
```

> **Python-only changes:** If you used `PYTHONPATH` or an editable install above,
> you can skip this step.

**Step 6 — (Optional) Reinstall the SGLang Python package**

Only needed if you prefer a non-editable install, or if you changed packaging
metadata and want to refresh the installed package:

```bash
cd /workspace/sglang/python
cp pyproject_cpu.toml pyproject.toml
unset CC CXX
export CXXFLAGS="-Wno-error"
uv pip install . --no-deps --index-strategy unsafe-best-match
```


## Launch the Server

Once the container is set up and kernels are compiled, launch the server:

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --attention-backend rvv \
    --device cpu \
    --dtype bfloat16 \
    --trust-remote-code \
    --mem-fraction-static 0.6 \
    --max-total-tokens 512 \
    --host 0.0.0.0 \
    --port 30000
```

Notes:

1. `SGLANG_USE_CPU_ENGINE` and `LD_PRELOAD` are already set inside the image —
   no additional exports are needed when using the Docker workflow.

2. `--mem-fraction-static` controls how much RAM is reserved for the KV cache.
   `--max-total-tokens` caps the total token budget. Reduce both if the server crashes
   with an out-of-memory error.

3. The server is ready when you see `The server is fired up and ready to roll!`

4. Once the server is running, open a second terminal and send requests — see
   [Send Inference Requests](#send-inference-requests) below.

### Quantization

The RVV backend supports W8A8 INT8 weight quantization:

| Mode | Extra flags | Weight memory |
|------|-------------|---------------|
| BF16 (default) | *(none)* | baseline |
| W8A8 INT8 | `--quantization w8a8_int8` | −50% |

Add the relevant flags to the `launch_server` command above. For example, to use
W8A8 quantization:

```bash
python3 -m sglang.launch_server \
    --model-path RedHatAI/Qwen2.5-1.5B-quantized.w8a8 \
    --attention-backend rvv \
    --device cpu \
    --dtype bfloat16 \
    --trust-remote-code \
    --quantization w8a8_int8 \
    --mem-fraction-static 0.6 \
    --max-total-tokens 512 \
    --host 0.0.0.0 \
    --port 30000
```

Notes:

1. **W8A8 INT8:** Requires a pre-quantized INT8 checkpoint with `weight_scale` tensors.
   Loading a plain BF16 checkpoint with `--quantization w8a8_int8` is **not supported**.

### Server Flags Reference

| Flag | Description |
|------|-------------|
| `--attention-backend rvv` | Use the hand-optimized RVV attention kernel |
| `--device cpu` | Run on CPU (RISC-V) |
| `--dtype bfloat16` | Use BF16 precision (recommended for best throughput) |
| `--mem-fraction-static` | Fraction of memory reserved for the KV cache |
| `--max-total-tokens` | Limit total token budget to fit within available RAM |
| `--quantization w8a8_int8` | Use pre-quantized INT8 weights (requires INT8 checkpoint with weight_scale) |

---

## Send Inference Requests

Once the server is running, query it from another terminal
(or from the host machine when using `--network host`).

**Using the OpenAI-compatible API:**

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is RISC-V?"}
    ],
    "max_tokens": 64,
    "temperature": 0.1
  }'
```

## Benchmarking

SGLang provides several official benchmark tools. Adjust `--max-total-tokens`
and `--mem-fraction-static` to match your board's available RAM.

### Online Serving Benchmark

Measures throughput and latency under a realistic request-rate scenario.
Start the server first, then in a separate terminal:

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url http://127.0.0.1:30000 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset-name random \
    --num-prompts 16 \
    --max-concurrency 2 \
    --request-rate 0.1 \
    --random-input-len 128 \
    --random-output-len 64
```

### Offline Throughput

Runs the engine in-process (no server required) and measures raw throughput:

```bash
python3 -m sglang.bench_offline_throughput \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --device cpu \
    --dtype bfloat16 \
    --attention-backend rvv \
    --dataset-name random \
    --num-prompts 16 \
    --random-input-len 128 \
    --random-output-len 64
```

### Static Batch Latency

Measures decode latency for a fixed static batch (no HTTP overhead):

```bash
python3 -m sglang.bench_one_batch \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --device cpu \
    --dtype bfloat16 \
    --attention-backend rvv \
    --batch 1 \
    --input-len 128 \
    --output-len 64
```

For all available options: `python3 -m sglang.bench_serving -h`, etc.

## Running Unit Tests

Verify the correctness of RVV kernels and backend wiring (GEMM, Decode, Extend, RoPE, Norm, registration, etc.).
Run the following directly inside the dev container:

```bash
cd /workspace/sglang

# (Optional) Only needed if you modified Python files in the mounted repo without reinstalling.
export PYTHONPATH=/workspace/sglang

/opt/.venv/bin/python -m unittest discover -s test/srt/cpu/rvv -p "test_*.py" -t . -v

# Run a specific test
/opt/.venv/bin/python -m unittest test.srt.cpu.rvv.test_rvv_gemm -v
```

Notes:

1. Run from the repository root (`/workspace/sglang`) so `-t .` resolves imports correctly.
2. The `test_*.py` pattern matches the full RVV unit test set under `test/srt/cpu/rvv/`,
   including files such as `test_rvv_backend_registration.py`, not just kernel-only tests.
3. If you already activated the virtual environment, `python` and `/opt/.venv/bin/python`
   are equivalent.

## Troubleshooting

**`sgl-kernel` build fails with compiler errors**

Make sure `CC`/`CXX` point to a valid Clang executable before running `uv pip install`.
The RVV intrinsic headers require Clang 19+; GCC does not support them.

```bash
command -v clang-19 || command -v clang
clang-19 --version 2>/dev/null || clang --version  # should show 19.x or later
```

**Server crashes with `munmap_chunk` or segfault**

This is typically caused by running out of memory. Try reducing `--max-total-tokens`
or `--mem-fraction-static`.

**Warning: `You are sending unauthenticated requests to the HF Hub...`**

Set a Hugging Face access token so downloads are authenticated and rate limits are higher.
Create a token at https://huggingface.co/settings/tokens and then set `HF_TOKEN`:

**Option A: Pass it into the container**

```bash
docker run -d \
    --name sglang-rvv-dev \
    --network host \
    -v $(pwd):/workspace/sglang \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e "HF_TOKEN=<your_token>" \
    sglang-rvv:latest \
    sleep infinity
```

**Option B: Export inside the running container**

```bash
export HF_TOKEN=<your_token>
```

**`hasattr(torch.ops.sgl_kernel, ...)` returns `False` unexpectedly**

This is a known limitation of `torch.ops`. Use a `try/except` block instead of `hasattr`:

```python
try:
    result = torch.ops.sgl_kernel.some_op(...)
except AttributeError:
    result = fallback_impl(...)
```

**Python changes are not reflected in the container**

Check where `sglang` is imported from:

```bash
python3 - <<'PY'
import sglang
print(sglang.__file__)
PY
```

If the path points to `/opt/.venv`, enable `PYTHONPATH` or use the editable install
in Step 4, or reinstall the package as shown in Step 6.
