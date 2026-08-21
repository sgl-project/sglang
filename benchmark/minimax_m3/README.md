# MiniMax-M3 FlashInfer MSA end-to-end gate

This gate compares SGLang's standalone `fmha_sm100` path with FlashInfer's
source-distributed MSA path on the same host, model, software environment, and
GPU clocks. It also records the Triton fallback as an optional diagnostic.

Use a 4x GB300 node with enough memory for the selected checkpoint. The formal
setup is `MiniMaxAI/MiniMax-M3-MXFP8` at TP4 with BF16 KV cache,
`--attention-backend fa4`, page size 128, CUDA Graph enabled, and no speculative
decoding. The FlashInfer checkout must contain the public
`flashinfer.msa_ops.{msa_sparse_attention,msa_sparse_decode_attention,msa_topk_select}`
API and its source files; install it into the same environment as SGLang. Do not
install a prebuilt JIT cache when validating a source export.

At TP4 the released model supplies the public API with these exact shapes:

- prefill: Q `[total_q, 16, 128]`, K/V `[num_pages, 1, 128, 128]`, sparse
  indices `[1, total_q, 16]`, cumulative Q lengths `[batch + 1]`, and page
  table `[batch, max_pages]`;
- decode: Q `[batch, 16, 128]`, the same paged K/V layout, sparse indices
  `[1, batch, 16]`, page table `[batch, max_pages]`, sequence lengths `[batch]`,
  and `seqlen_q=1`.

The public source implementation is selected only for top-k 16, the released
MiniMax-M3 value. Each sparse layer receives a distinct persistent public
workspace for each CUDA Graph capture. Q, sparse indices, and length metadata
are copied into capture-stable buffers; the page table is refreshed in place.
The warmup stream is synchronized before capture, and replay invokes no Python
planning API or private plan tuple. Explicit provider selection fails closed on
a runtime rejection, so an A/B run cannot silently turn into a Triton fallback.

Run the two servers sequentially on the same idle node. Keep every launch flag
identical except the provider. The compatibility baseline needs its existing
decode-under-graph opt-in; setting it for both launches keeps the comparison
explicit and symmetric. This gate disables the dynamic-request prefill BCG so
both providers exercise their own eager prefill while decode remains captured:

```bash
COMMON_ARGS=(
  --model-path MiniMaxAI/MiniMax-M3-MXFP8
  --trust-remote-code
  --reasoning-parser auto
  --tool-call-parser auto
  --tp 4
  --attention-backend fa4
  --page-size 128
  --moe-runner-backend deep_gemm
  --chunked-prefill-size 8192
  --mem-fraction-static 0.75
  --disable-prefill-cuda-graph
)

# baseline
SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH=1 \
SGLANG_MINIMAX_MSA_BACKEND=fmha_sm100 \
  python -m sglang.launch_server "${COMMON_ARGS[@]}"

# candidate
SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH=1 \
SGLANG_MINIMAX_MSA_BACKEND=flashinfer \
  python -m sglang.launch_server "${COMMON_ARGS[@]}"

# optional fallback reference
SGLANG_DISABLE_MSA=1 python -m sglang.launch_server "${COMMON_ARGS[@]}"
```

Before any measured run, freeze both accuracy inputs. Download GPQA-Diamond
once, then build the LongBench subset with the candidate model's tokenizer:

```bash
curl -fL \
  https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv \
  -o /shared/eval/gpqa_diamond.csv

python benchmark/minimax_m3/build_longbench_subset.py \
  --model MiniMaxAI/MiniMax-M3-MXFP8 \
  --num-examples 100 --min-tokens 32768 --max-tokens 524288 \
  --output /shared/eval/longbench_v2_m3_100_min32k.json
```

The builder maps LongBench-v2's six human-readable `domain` labels to the
canonical SGLang task-category names before balancing the subset. The manifest
records that mapping, the per-category counts, and the tokenizer-observed
minimum and maximum prompt lengths.

The preflight is offline and fail-closed. When `--output` is omitted it writes
no evidence file, although the real plan-ABI probe may populate the configured
session JIT cache. It resolves
the model from the local Hugging Face cache (or a local path), checks every
indexed weight shard, loads the tokenizer locally, verifies both dataset hashes
and row counts, checks the exact clean FlashInfer source HEAD and installed
public API, checks the standalone baseline import, verifies this SGLang checkout
is the one Python imports, executes the baseline's real decode-plan ABI, and
requires exactly four visible compute-capability 10.3 GPUs. The compatibility
image carries `apache-tvm-ffi==0.1.9`; preserve that runtime for the A/B instead
of shadowing it with a second FFI DSO in the session environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python benchmark/minimax_m3/probe_msa_e2e_dependencies.py \
  --model MiniMaxAI/MiniMax-M3-MXFP8 \
  --longbench-subset /shared/eval/longbench_v2_m3_100_min32k.json \
  --gpqa-dataset /shared/eval/gpqa_diamond.csv \
  --flashinfer-source-dir "${FLASHINFER_SOURCE_DIR}" \
  --expected-flashinfer-head "${FLASHINFER_HEAD}" \
  --expected-tvm-ffi-version 0.1.9
```

These smaller probes are also useful before requesting the allocation; none
downloads or modifies a cache:

```bash
git -C "${FLASHINFER_SOURCE_DIR}" rev-parse HEAD
git -C "${FLASHINFER_SOURCE_DIR}" status --short
test -r /shared/eval/gpqa_diamond.csv
test -r /shared/eval/longbench_v2_m3_100_min32k.json
test -r /shared/eval/longbench_v2_m3_100_min32k.json.manifest.json
```

For the publishable gate, use the automated driver inside an already-created,
exclusive 4x GB300 Slurm allocation. This command is suitable as the payload of
an `sbatch --wrap` or session-specific job script:

```bash
export MODEL=MiniMaxAI/MiniMax-M3-MXFP8
export LONGBENCH_SUBSET=/shared/eval/longbench_v2_m3_100_min32k.json
export GPQA_DATASET=/shared/eval/gpqa_diamond.csv
export FLASHINFER_SOURCE_DIR=/workspace/flashinfer
export FLASHINFER_HEAD=<exact-final-source-commit>
export OUTPUT_ROOT=/shared/results/msa_gb300_tp4_$(date -u +%Y%m%dT%H%M%SZ)
export EXPECTED_TVM_FFI_VERSION=0.1.9
export GPQA_SCORE_TOLERANCE=0
export LONGBENCH_SCORE_TOLERANCE=0.01
export MINFER_FMHA_CACHE_DIR=/workspace/run/cache/fmha_sm100
export BASELINE_FMHA_PRECOMPILE_RECEIPT=/shared/results/fmha_sm100_precompile.json
export SERVER_TIMEOUT=7200
bash benchmark/minimax_m3/run_msa_ab_gb300.sh
```

Before the first server starts, the driver serially compiles and validates the
BF16 sparse-paged `fmha_sm100` variants reachable by this TP4 gate, plus its
plan, reduction, and sparse-top-k modules. The baseline package only protects
JIT compilation within one process; serial precompilation prevents four TP
workers from racing on the same shared `.so` while preserving one identical
completed cache for all six server starts.

For a fresh run, the driver refuses an existing output root and runs exactly
three complete repetitions in `baseline,candidate`, `candidate,baseline`, then
`baseline,candidate` order. Every backend gets a fresh server. The driver waits
for `/health_generate`, verifies one startup-log line contains the requested
`main_attn`, `msa_decode=True`, `msa_owns_decode=True`, and
`decode_cuda_graph=True`, sends an unmeasured 8K/1K warmup, and only then starts
the full gate. Both warmup and serving sweeps use the fixed-seed, offline
`random-ids` generator and send token IDs directly, so the 8K input length is
exact and they never depend on an implicit dataset download. It stops the
server between providers and fails if the old server still owns the port.
If an allocation ends between complete repetitions, set `START_REPETITION` to
2 or 3 and reuse the output root. Resume is fail-closed: immutable manifest
inputs and every completed pair are revalidated, while an incomplete target
repetition is never overwritten.

Each repetition is compared independently. `summary.json` then reports the
three raw values, backend median, gain computed from backend medians, and median
paired gain for every concurrency and metric. It also rejects provider-order
drift, dataset-hash drift, and any visible temperature-zero fixed answer that
changes across the six server runs. Hidden reasoning text and its response hash
remain in the evidence for auditing, but are not an equality gate: equivalent
exact answers can legitimately use different reasoning traces. By default,
candidate median output throughput may not regress at any concurrency.

For a single diagnostic run, start one server manually and run:

```bash
BASE_URL=http://127.0.0.1:30000 \
MODEL=MiniMaxAI/MiniMax-M3-MXFP8 \
LABEL=external \
OUTPUT_DIR=/shared/results/msa_external \
LONGBENCH_SUBSET=/shared/eval/longbench_v2_m3_100_min32k.json \
GPQA_DATASET=/shared/eval/gpqa_diamond.csv \
bash benchmark/minimax_m3/run_msa_gate.sh
```

Use `LABEL=flashinfer` and a new output directory for the candidate, then check
the accuracy contract:

```bash
python benchmark/minimax_m3/compare_msa_gate.py \
  --baseline-dir /shared/results/msa_external \
  --candidate-dir /shared/results/msa_flashinfer \
  --gpqa-score-tolerance 0 \
  --longbench-score-tolerance 0.01
```

The mandatory accuracy gates are exact temperature-zero output parity for the
fixed probes (with tokenizer-measured 32K and 64K prompt lengths), zero score
tolerance on all 198 GPQA-Diamond questions, and an explicit one-answer (0.01)
noninferiority margin on the deterministic 100-example category-balanced
LongBench-v2 subset whose prompts are 32K--512K tokens. Both the per-pair checks
and the three-run backend medians enforce those margins. The runner requires the
two margins as inputs and records them in its manifests and comparison JSON.
GPQA is materially harder than saturated GSM8K; LongBench directly exercises
MSA's long-context page selection and decode replay without admitting prompts
beyond the model's serving envelope.

The single-pair comparison command also reports fractional speedup for
request/output-token throughput and median/p99 TTFT and inter-token latency. Pass
`--min-output-throughput-gain 0` to make non-regression at every concurrency a
hard gate. The automated three-repetition gate applies non-regression to the
backend medians instead, which is less sensitive to one noisy pair. Keep
clocks/power limits fixed, and reject evidence with errors, retries, compilation
during a measured serving interval, or thermal throttling. The gate records the
server-log byte offsets bracketing every serving interval to make that review
exact rather than timestamp-based.
