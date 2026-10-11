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

SGLang now owns the missing prefill persistence in
`python/sglang/srt/layers/attention/linear/gdn_cache.py`. **No FlashInfer package
patch is required.** During init it restores exported callables into FlashInfer's
existing process-local cache, runs the ordinary preparation calls, then exports
new cache entries with CuTe's `export_to_c` API. Restart loading uses
`cute.runtime.load_module`. FlashInfer source and function identities remain
unchanged; its normal compilation path still performs SM12x PTX validation.

The adapter uses the private `_in_mem_compile_cache` explicit-key contract tested
with FlashInfer 0.7.0.post1 / CuTe 4.8.0. It recognizes GDN TVM-FFI keys only.
Unsupported entries are skipped; failed exports or incompatible dependencies
warn and fall back to ordinary preparation, so a future FlashInfer version needs a compatibility
check before assuming compile-free restarts. It does not hook arbitrary runtime
compilation or persist specializations first encountered after init.

Objects and a JSON key manifest live under `$SGLANG_CACHE_DIR/gdn_prefill/`.
The directory fingerprint includes architecture, GDN and adapter source hashes,
compiler-stack versions, Python ABI, FlashInfer, Torch, TVM FFI and CUDA bindings.
No pickle or imports from manifest contents are used. Exports and manifest
updates use a file lock and atomic replacement; corrupt entries fall back to
compilation and are repaired after successful preparation.
`FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` disables this persistence.

Keep the same `SGLANG_CACHE_DIR` and enable `SGLANG_PREPARE_LINEAR_ATTN=1` on
**every startup**. First startup compiles/tunes and writes disk artifacts;
later startups load them and prepare process-local handles. The previous
companion dependency patch has been removed from this branch.

On RTX5090, T256 init took 9.295 s cold and 0.158 s after restart
(median of 3 fresh processes); first post-init wall time was 0.621 ms versus
0.214 ms steady. Five objects totaling 1,144,920 bytes were saved.
Separate restarts with unchanged FlashInfer passed T256 and T4096
numerical checks while CuTe compilation and Triton lowering/autotuning were
forbidden. Tests also asserted that FlashInfer's cache source bytes and function
identities were unchanged. This verifies the measured operator paths, not a
complete server startup, Hopper/B200, or first-call equality with steady state.

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

For a strict restart check, first seed the chosen operation with `--prepare`,
then run it in a new process with `--prepare --forbid-compilation --diagnose`.
The test-only flag makes CuTe compilation/build and Triton lowering/autotuning
raise immediately; loading already compiled artifacts is still allowed. These
guards modify compiler entry points only inside the benchmark process. The
production SGLang cache adapter never replaces those functions.
