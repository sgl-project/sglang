# Eager linear attention initialization

Opt-in startup preparation runs after `ModelRunner.init_attention_backends`
constructs the backends, before graphs or user forwards. It calls only the
selected attention operators with private scratch state. It **does execute
kernels during init**; it is not an AOT-only or zero-launch preparation API.

```bash
export SGLANG_PREPARE_LINEAR_ATTN=1
export SGLANG_LINEAR_ATTN_PREPARE_PREFILL_TOKENS=256,4096
export SGLANG_LINEAR_ATTN_PREPARE_DECODE_BATCH_SIZES=1,8
export SGLANG_CACHE_DIR=/path/to/persistent/sglang-cache
```

The flag defaults to off. Empty size lists skip a mode. Sizes must be positive.
The initial implementation prepares Triton KDA prefill (including tracked state)
and FlashInfer GDN decode/prefill (including checkpoint outputs when applicable).
Other kernel/mode combinations log an unsupported warning and return zero cases.
It does not prepare convolution, projections, dense attention, speculative verify,
or backend-specific fused overrides outside the selected kernel dispatcher.

Preparation uses actual per-rank head dimensions, dtype, parameters and state
strides. Identical signatures on the same kernel object are prepared once across
model layers. Serving state is never passed to a mutating kernel. Checkpoint
interval comes from the selected backend. Larger batch/length/layout combinations
or different mode flags can introduce additional specializations; the configured
sizes are coverage, not a promise that every request is warmed.

The two active KDA autotuners now honor FLA's existing `FLA_CACHE_RESULTS=1`
default. Keep the Triton cache across processes. Each new process still has to
load its own modules/handles. FlashInfer cache behavior depends on the backend:
in unmodified 0.7.0.post1, GDN decode has a disk cache, but the SM120 GDN
prefill `cached_compile` path has only an in-process dictionary. Startup
preparation alone cannot fix that: it would recompile on every restart.

A companion FlashInfer patch is provided at
`patches/flashinfer-0.7.0.post1-gdn-disk-cache.patch`. It routes GDN's explicit
compile-key cache through FlashInfer's existing `JitSpecCuteDsl` object export /
load implementation. It retains SM12x PTX validation before export and uses
specialization, architecture, source hashes, compiler-stack versions, Python ABI,
Torch, TVM FFI and CUDA bindings versions to invalidate artifacts. Existing file
locks and atomic export are reused. Non-GDN and non-TVM-FFI calls keep the original
behavior; `FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` disables persistence.

Apply the companion patch **inside the venv running SGLang** (FlashInfer
0.7.0.post1 only). From the SGLang repository, after activating that venv:

```bash
FI_SITE=$(python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("flashinfer").origin).parent.parent)')
patch --dry-run -d "$FI_SITE" -p1 < benchmark/bench_linear_attention/patches/flashinfer-0.7.0.post1-gdn-disk-cache.patch
patch -d "$FI_SITE" -p1 < benchmark/bench_linear_attention/patches/flashinfer-0.7.0.post1-gdn-disk-cache.patch
export FLASHINFER_WORKSPACE_BASE=/path/to/persistent/flashinfer-cache
```

Do not reapply to an already patched environment. A package upgrade can replace
the patched file; this companion change is not an upstream release. The SGLang
flag controls init preparation; the dependency patch provides disk persistence.
Keep both the SGLang/Triton and FlashInfer cache directories between restarts.

With the companion patch on RTX5090, GDN prefill T256 preparation fell from a
9.406 s cold start to a 0.154 s restart median (3 fresh interpreters). Restart
first attention wall time was 0.554 ms, steady 0.216 ms. Five `.o` artifacts
(total 1,145,320 bytes) were written. Separate restart tests prohibited CuTe
compilation and Triton lowering/autotuning and still passed numerical checks
for T256 and T4096. This proves disk reuse for those cases, not full-server
coverage or first-call equality with steady state.

For a server-level eager test, configure both graph phases as disabled:

```bash
--cuda-graph-config '{"prefill":{"backend":"disabled"},"decode":{"backend":"disabled"}}'
```

## Isolated measurement

`bench_first_call.py` imports normal SGLang code and uses two real
`RadixLinearAttention` modules as model metadata. It invokes the production
preparation hook, then times one real kernel wrapper. It does not launch a server
or claim complete model initialization has been tested. All imports, context
creation, input construction and reference calculations are outside the timing.
Init and the first eager operator call are reported separately. No graph is used.

Run from the repository with a suitable venv, setting `PYTHONPATH=python` if the
source tree is not installed. Every invocation below is a separate process. Keep
the same cache directory; use new report filenames on reruns.

```bash
export SGLANG_CACHE_DIR=/path/to/experiment/cache
export FLA_CACHE_RESULTS=1
# Explicitly disable the global override to test SGLang's decorator fix.
export TRITON_CACHE_AUTOTUNING=0

# Process 1: build/tune caches in init.
python benchmark/bench_linear_attention/bench_first_call.py \
  --op kda-prefill --size 256 --prepare --report seed.json

# Process 2: reuse cache; time init separately from first attention invocation.
python benchmark/bench_linear_attention/bench_first_call.py \
  --op kda-prefill --size 256 --prepare --report reuse-on.json

# Control: same disk cache, no operator preparation.
python benchmark/bench_linear_attention/bench_first_call.py \
  --op kda-prefill --size 256 --report reuse-off.json
```

Also supported: `--op gdn-decode --size 1` and `--op gdn-prefill --size 256`.
`--diagnose` separately counts Triton cache misses/benchmarks and CuTe compile
requests by phase. `--qualify` uses poisoned allocations and a single measured
invocation. All runs check the first output/state against an independent FP64
recurrence, guard tensor boundaries, and assert preparation did not alter the
serving state. The benchmark currently uses BF16, H=8, K=V=128 and FP32 state.

Wall time includes Python dispatch and synchronization. CUDA event spans include
CPU enqueue gaps, so they are not pure GPU execution times. Do not include
diagnostic or sanitizer runs in latency summaries, and do not call the first
post-init invocation the first-ever kernel execution.
