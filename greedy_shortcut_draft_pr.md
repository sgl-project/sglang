# Draft PR: AMD greedy LM-head AG+argmax shortcut with EAGLE compatibility

## Summary

This PR adds an AMD/HIP-only fast path for greedy decode when the LM head is tensor-parallel sharded. Instead of materializing full-vocab logits via TP
all-gather and then running `torch.argmax`, each rank computes its local `max/argmax`, all-gathers only `(max_value, local_index)` pairs, and reconstructs the global token id.

The route is opt-in via:

```bash
SGLANG_AITER_AG_ARGMAX_SHORTCUT=1
SGLANG_AITER_AG_ARGMAX_SHORTCUT_MIN_M=2
```

Non-HIP platforms are explicitly gated out and continue using the existing full-logits path.

## What changed

- Added `_fused_greedy_argmax_across_tp(local_logits)` in the logits processor.
- Added `next_token_ids_shortcut` to `LogitsProcessorOutput`.
- Added sampler support for consuming precomputed greedy token ids.
- Preserved CUDA graph replay plumbing for `next_token_ids_shortcut`.
- Made EAGLE compatible:
  - Draft inputs keep full logits because draft generation needs `softmax`.
  - AMD/HIP target verification can consume `next_token_ids_shortcut` for greedy verification.
  - Non-HIP EAGLE paths keep the old full-logits assumptions.
- Added a torchrun-able correctness and benchmark test:
  - `test/srt/test_greedy_argmax_shortcut.py`

## Serving results

Server command:

```bash
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
  SGLANG_ENABLE_SPEC_V2=1 \
  ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
  SGLANG_AITER_FP8_PREFILL_ATTN=1 \
  SGLANG_AITER_MLA_PERSIST=1 \
  AITER_MXFP4_MOE_SF=1 \
  SGLANG_USE_AITER=1 \
  SGLANG_INT4_WEIGHT=0 \
  SGLANG_MOE_PADDING=1 \
  SGLANG_SET_CPU_AFFINITY=1 \
  SGLANG_ROCM_FUSED_DECODE_MLA=1 \
  SGLANG_USE_ROCM700A=1 \
  SGLANG_DISABLE_FUSED_AR_MXFP4_QUANT=false \
  python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-MXFP4 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --mem-fraction-static 0.9 \
    --chunked-prefill-size 131072 \
    --attention-backend aiter \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --max-running-requests 32 \
    --context-length 200000 \
    --kv-cache-dtype fp8_e4m3 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
```
## Correctness gates

GSM8K on `amd/DeepSeek-R1-MXFP4`, TP=4, EAGLE enabled, AITER route:
```bash
python3 benchmark/gsm8k/bench_sglang.py \
  --num-questions 1319 \
  --parallel 1319 \
  --num-shots 5 \
  --port 8000
```
Baseline:

```text
run 1: 0.952
run 2: 0.943
run 3: 0.937
```

Optimized:

```text
run 1: 0.948
run 2: 0.945
run 3: 0.944
```


### `max-concurrency=2`

Workload:

```text
random-input-len=70000
random-output-len=500
num-prompts=16
max-concurrency=2
```

Result:

```text
total token throughput: 12252.08 tok/s vs baseline 11280.16 tok/s (+8.6%)
median TPOT:            14.04 ms vs baseline 15.80 ms
```

### `max-concurrency=4`

Workload:

```text
random-input-len=70000
random-output-len=200
num-prompts=32
max-concurrency=4
```

Result:

```text
total token throughput: 24363.59 tok/s vs baseline 23657.53 tok/s (+3.0%)
median TPOT:            32.21 ms vs baseline 30.37 ms
```

The `mc4` throughput improved while median TPOT regressed slightly; this should be called out as a tradeoff in review.

## Shared-prefix cache bench

Quick sweep using `benchmark/hicache/bench_warm_cache.py`:

```text
pcts=0,60,90,99
total_tokens=70000
output_len=500
num_prompts=16
max_concurrency=2
```

Artifacts:

```text
benchmark/hicache/bench_warm_cache.py
```

Headline:

```text
0% shared prefix:  TPM/GPU -1.80%, median TPOT -0.46%
60% shared prefix: TPM/GPU +1.84%, median TPOT +11.71%
90% shared prefix: TPM/GPU +2.71%, median TPOT +0.24%
99% shared prefix: TPM/GPU +0.98%, median TPOT +0.32%
```

## Unit/benchmark test

Correctness only:

```bash
SGLANG_AITER_AG_ARGMAX_SHORTCUT=1 \
PYTHONPATH=/sgl-workspace/sglang/python \
torchrun --nproc_per_node=4 --master_port=29510 \
  test/srt/test_greedy_argmax_shortcut.py
```

Correctness plus benchmark:

```bash
SGLANG_AITER_AG_ARGMAX_SHORTCUT=1 \
PYTHONPATH=/sgl-workspace/sglang/python \
torchrun --nproc_per_node=4 --master_port=29510 \
  test/srt/test_greedy_argmax_shortcut.py \
  --benchmark \
  --m-values 1,2,8,16,32,64,128 \
  --v-local 31040 \
  --csv-out /tmp/greedy_argmax_shortcut_bench.csv
```
```
PASS correctness: M=[1, 2, 8, 16, 32, 64, 128], V_local=31040, TP=4, dtype=torch.bfloat16
     M   base_eager     shortcut   eager_x   base_graph   shortcut_g   graph_x
     1       0.0624       0.1043     0.60x       0.0520       0.0431     1.21x
     2       0.0722       0.1045     0.69x       0.0651       0.0434     1.50x
     8       0.0785       0.1050     0.75x       0.0715       0.0433     1.65x
    16       0.0898       0.1049     0.86x       0.0829       0.0442     1.87x
    32       0.1146       0.1151     1.00x       0.1073       0.0438     2.45x
    64       0.1563       0.1031     1.52x       0.1508       0.0449     3.36x
   128       0.2400       0.1021     2.35x       0.2340       0.0454     5.16x
```

The test skips shortcut execution on non-HIP devices so other hardware routes remain untouched.

## Risk and review notes

- This is AMD/HIP-only and opt-in.
- Draft-side EAGLE still requires full logits for `softmax`; the shortcut is gated off for speculative draft inputs.
- Target-side EAGLE greedy verification can consume `next_token_ids_shortcut`.
- Non-HIP EAGLE and sampler paths keep the original full-logits behavior.
- The route intentionally falls back for logprobs, full logits, logit bias, vocab masks, grammars, custom logit processors, penalties, softcapping, LoRA/AMX wrapping, DP-attention gather, and attention-TP-group gather.
