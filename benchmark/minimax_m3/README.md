# MiniMax-M3 FlashInfer MSA end-to-end gate

This gate compares SGLang's standalone `fmha_sm100` path with FlashInfer's
source-distributed MSA path on the same host, model, software environment, and
GPU clocks. It also records the Triton fallback as an optional diagnostic.

Use a Blackwell node with enough memory for the selected checkpoint. The
standard B200 setup is `MiniMaxAI/MiniMax-M3-MXFP8` at TP4 with BF16 KV cache,
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
planning API or private plan tuple.

Run the two servers sequentially on the same idle node. Keep every launch flag
identical except the provider. The compatibility baseline needs its existing
decode-under-graph opt-in; setting it for both launches keeps the comparison
explicit and symmetric:

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

Before either measured run, wait for model load and CUDA Graph capture, send one
unmeasured 8K/1K request, and verify the startup log says `main_attn=...` for the
intended path. Freeze the LongBench subset once with the candidate model's
tokenizer:

```bash
python benchmark/minimax_m3/build_longbench_subset.py \
  --model MiniMaxAI/MiniMax-M3-MXFP8 \
  --num-examples 100 --min-tokens 32768 \
  --output /shared/eval/longbench_v2_m3_100_min32k.json
```

Then run the gate once per server:

```bash
BASE_URL=http://127.0.0.1:30000 \
MODEL=MiniMaxAI/MiniMax-M3-MXFP8 \
LABEL=external \
OUTPUT_DIR=/shared/results/msa_external \
LONGBENCH_SUBSET=/shared/eval/longbench_v2_m3_100_min32k.json \
bash benchmark/minimax_m3/run_msa_gate.sh
```

Use `LABEL=flashinfer` and a new output directory for the candidate, then check
the accuracy contract:

```bash
python benchmark/minimax_m3/compare_msa_gate.py \
  --baseline-dir /shared/results/msa_external \
  --candidate-dir /shared/results/msa_flashinfer
```

The mandatory accuracy gates are exact temperature-zero output parity for the
fixed probes, no regression on all 198 GPQA-Diamond questions, and no regression
on the deterministic 100-example category-balanced LongBench-v2 subset whose
prompts are at least 32K tokens. GPQA is materially harder than saturated GSM8K;
LongBench directly exercises MSA's long-context page selection and decode replay.

The comparison command also reports fractional speedup for request/output-token
throughput and median/p99 TTFT and inter-token latency. Pass
`--min-output-throughput-gain 0` to make non-regression at every concurrency a
hard gate. For a publishable result, report the median of three complete runs
for each backend and concurrency. Alternate backend order across repetitions,
keep clocks/power limits fixed, flush the radix cache, and reject any run with
errors, retries, JIT compilation, or thermal throttling.
