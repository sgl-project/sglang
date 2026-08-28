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

For publishable evidence, use the one canonical driver inside a fresh,
exclusive 4x GB300 allocation. Freeze the SGLang and FlashInfer commit and tree,
the full checkpoint file manifest and aggregate (with the config hash as an
additional fast check), and both evaluation inputs before starting. The three modes
must use distinct output and cache roots:

```bash
COMMON=(
  --model /model
  --gpqa-dataset /shared/eval/gpqa_diamond.csv
  --longbench-subset /shared/eval/longbench_v2_m3_100_min32k.json
  --flashinfer-source-dir /workspace/flashinfer
  --expected-flashinfer-head "${FLASHINFER_HEAD}"
  --expected-flashinfer-tree "${FLASHINFER_TREE}"
  --sglang-source-dir /workspace/sglang
  --expected-sglang-head "${SGLANG_HEAD}"
  --expected-sglang-tree "${SGLANG_TREE}"
  --expected-model-config-sha256 "${MODEL_CONFIG_SHA256}"
  --model-manifest /shared/input/model-checkpoint-manifest.json
  --expected-model-manifest-sha256 "${MODEL_MANIFEST_SHA256}"
  --expected-model-aggregate-sha256 "${MODEL_AGGREGATE_SHA256}"
  --expected-gpqa-sha256 "${GPQA_SHA256}"
  --expected-longbench-sha256 "${LONGBENCH_SHA256}"
  --expected-longbench-manifest-sha256 "${LONGBENCH_MANIFEST_SHA256}"
  --server-timeout 7200
)

python benchmark/minimax_m3/run_msa_formal_v2.py \
  "${COMMON[@]}" --mode accuracy \
  --output-root /shared/results/msa_accuracy \
  --cache-root /workspace/cache/msa_accuracy

python benchmark/minimax_m3/run_msa_formal_v2.py \
  "${COMMON[@]}" --mode external-speed \
  --output-root /shared/results/msa_external_speed \
  --cache-root /workspace/cache/msa_external_speed \
  --min-median-throughput-gain 0

python benchmark/minimax_m3/run_msa_formal_v2.py \
  "${COMMON[@]}" --mode triton-speed \
  --output-root /shared/results/msa_triton_speed \
  --cache-root /workspace/cache/msa_triton_speed \
  --min-median-throughput-gain 0
```

Accuracy runs exactly three fresh, alternating pairs in the order
`external,flashinfer`, `flashinfer,external`, `external,flashinfer`. Each speed
mode runs exactly one fresh pair: `external,flashinfer` for external speed and
`triton,flashinfer` for the Triton reference. There are no `rep02` or `rep03`
speed runs. Every arm gets a new server and isolated JIT/cache directories. The
driver verifies the selected route, sends one unmeasured fixed-seed 8K/1K
warmup, brackets the measured server log by byte offset, and rejects port reuse,
JIT activity, errors, retries, timeouts, client exhaustion, thermal throttling,
or an unexpected count of successful requests. It never resumes into an
existing result root.
For the loopback OpenAI-compatible evaluator client, the runner supplies the
non-secret dummy key `EMPTY` only when `OPENAI_API_KEY` is absent; an explicitly
provided value is preserved.
Before launching a server, every allocation re-hashes the complete checkpoint,
requires its exact file set, and compares every file plus the aggregate against
the frozen manifest; a config hash alone is not a checkpoint identity.

Build the Python environment once as an immutable input, not independently in
the three formal allocations. Resolve the complete transitive wheel closure for
the target Python ABI and platform into a session-local wheelhouse, record the
exact filename, size, and SHA-256 of every wheel plus an aggregate, and reject
missing or extra files. Each allocation must verify that manifest, install only
those explicit wheel paths with `PIP_NO_INDEX=1`, `--no-index`, and `--no-deps`,
then run the dependency/import preflight. Do not use a network package index in
a formal allocation.

Accuracy and speed are separate experiments. Accuracy runs exact visible-answer
probes at short, 32K, and 64K lengths, all 198 GPQA-Diamond questions at one
client thread, and the deterministic 100-example LongBench-v2 subset at one
thread. Both every pair and the three-run aggregate must stay within one GPQA
question and 0.02 LongBench score. Private GPQA responses are mode `0600`;
public per-example evidence contains hashes rather than response text. GPQA is
materially harder than saturated GSM8K, while LongBench directly exercises
long-context page selection and decode replay.

The two speed modes send no accuracy requests. Each arm runs exactly 256 native
SGLang `/generate` requests at concurrency 1, 8, 32, and 128, with fixed-seed
`random-ids`, 8192 input tokens, 1024 output tokens, range ratio 1, and implicit
benchmark warmups disabled. The consumer rejects duplicate JSONL records,
partial or failed request counts, workload drift, non-finite values, and a
negative median output-throughput gain at any concurrency. Keep clocks and
power limits fixed and retain the complete receipts and logs for review.

Run the executable fail-closed contract without a GPU with:

```bash
python benchmark/minimax_m3/run_msa_formal_v2.py --test-only
```
