# AGENTS.md

## Cursor Cloud specific instructions

This section captures non-obvious, durable context for running SGLang in the Cursor
Cloud VM. It is **not** a general contributor guide — for tests/CI see
[`test/README.md`](test/README.md) and the skills under `.claude/skills/`; for the
CPU path see [`docs/docs/hardware-platforms/cpu_server.mdx`](docs/docs/hardware-platforms/cpu_server.mdx).

### This VM has no NVIDIA GPU → use the CPU build, not the CUDA build

- The default `python/pyproject.toml` is the **CUDA** build. Do **not** try to install it
  here. Besides needing a GPU, it is currently uninstallable anywhere: `flashinfer-python`
  pins `cuda-tile==1.6.0rc6`, a version that is not published on `pypi.nvidia.com`
  (only `1.0.0`/`1.0.0rc6` exist), so resolution fails on any machine.
- The environment is set up for the **CPU** path (`python/pyproject_cpu.toml`), which is
  what `docker/xeon.Dockerfile` uses. The VM CPU is an Intel Xeon **with AMX**
  (`amx_tile`/`amx_bf16`/`amx_int8`), so SGLang's `intel_amx` CPU backend runs real inference.

### Environment layout

- A Python **3.12** virtualenv lives at `/workspace/.venv` (created with `uv`, which is at
  `~/.local/bin/uv`). `torch` is the CPU wheel (`torch==2.12.0+cpu`).
- `sglang` is installed **editable**, so Python source edits are picked up without reinstall.
- The Rust default toolchain is pinned to **1.92** (`rustup default 1.92`); the workspace
  needs edition 2024, which the image's older 1.83 cannot build. `setuptools-rust` uses the
  *default* toolchain, so the default must stay 1.92.
- The compiled CPU kernel `sglang-kernel-cpu` is installed **non-editable**. If you change
  C++ under `python/sglang/kernels/aot/csrc/cpu`, rebuild it manually (see below); the
  startup update script does not rebuild it.
- The dependency install temporarily swaps `python/pyproject_cpu.toml` over
  `python/pyproject.toml` and restores it. Keep `python/pyproject.toml` unmodified in git.

### Running things (activate the venv first)

```bash
source /workspace/.venv/bin/activate
```

- **Lint:** `pre-commit run --files <path...>` (pre-commit is installed in the venv).
  Rust hooks need `cargo`/`clippy` on PATH (`export PATH="/usr/local/cargo/bin:$PATH"`).
- **Tests (CPU):** run a file directly, e.g.
  `python test/registered/unit/sampling/test_sampling_params.py`, or a suite slice:
  `cd test && SGLANG_IS_IN_CI=true python run_suite.py --hw cpu --suite base-a-test-cpu --auto-partition-id 0 --auto-partition-size 40`.
  The full suite is long (CI partitions it). Note `test/registered/rust/test_cargo_workspace.py`
  compiles the whole Rust workspace and currently fails on a transitive crate (`esaxx-rs`)
  C++ build — that is an environment limitation, unrelated to the Python dev setup.
- **Serve a model on CPU** — these env vars are required (without them the CPU engine
  won't start correctly):

```bash
export SGLANG_USE_CPU_ENGINE=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/workspace/.venv/lib/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:/usr/lib/x86_64-linux-gnu/libtbbmalloc.so.2
export HF_HOME=/workspace/.hf-cache
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --device cpu --disable-overlap-schedule --tp 1 --host 127.0.0.1 --port 30000
# then: curl http://127.0.0.1:30000/generate -d '{"text":"Hello","sampling_params":{"max_new_tokens":16}}'
```

  Use small, ungated models (e.g. `Qwen/Qwen2.5-0.5B-Instruct`). `meta-llama/*` defaults in
  tests are gated and need `HF_TOKEN`.

### Rebuilding the CPU kernel (only when C++ changes)

```bash
cd python/sglang/kernels/aot
cp pyproject.toml /tmp/aot.bak && cp pyproject_cpu.toml pyproject.toml
uv pip install --python /workspace/.venv/bin/python . \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
cp /tmp/aot.bak pyproject.toml
```

For interactive shells, `~/.bashrc` already activates `/workspace/.venv` and exports the
CPU serving env vars above.
