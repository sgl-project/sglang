# Experimental SDAR refresh and denoise pairing

SDAR denoising predicts tokens from masked or partially committed inputs.
The resulting KV tensors describe those inputs, not the newly selected tokens.
A completed block therefore needs a refresh before the next block uses its KV.

This prototype combines that refresh with the next block's first denoising pass.
Linear layers process `[completed block A, masked block B]` together.
Within each attention layer, two existing FlashInfer calls preserve the dependency:

1. Process A against the cached prefix and A. Write A's refreshed KV.
2. Process B against the prefix, refreshed A, and B.

This preserves block-causal attention without a new attention kernel.
It retains both blocks' arithmetic. Larger linear operations can reduce repeated
weight reads and launch overhead. The benefit depends on batch size and hardware.

The backend supports dense SDAR, LowConfidence decoding, BF16 model and KV,
one GPU, full attention, page size one, and the stock ragged FlashInfer path.
It is not registered as a server backend. The benchmark owns block progression.
The production scheduler, request admission, streaming, and cancellation are unchanged.

## Compare separate and paired execution

Supply a JSON list of 128 token IDs from a fixed prompt. Use the same IDs for
every comparison. Set the checkpoint revision explicitly when downloading weights.

```bash
PYTHONPATH=python python benchmark/dllm/bench_refresh_denoise_pair.py \
  --model-path /path/to/SDAR-8B-Chat-b16 --block-size 16 \
  --prompt-ids prompt-128.json --concurrency 1 --trials 5 \
  --profile --output results/sdar-b16-n1
```

Repeat for B32 and fixed batch sizes 4 and 16. Use `--block-size 0` with
Qwen3-8B for the AR control. Each output directory must be new.

The benchmark tests confidence thresholds 0, 0.5, and 0.95. Both SDAR variants
use native LowConfidence selection and an identical completion check.
Each request generates 256 tokens. Both variants retain the terminal refresh.
Both variants use captured model CUDA Graphs. Planning and token selection
remain inside each measured generation interval. Capture and warmup are excluded.

Before timing, the benchmark checks logits, retained KV, selected tokens, and
graph replay against eager execution. BF16 tolerances are fixed in the harness.
Each paired timing result is admitted only when full output token IDs match.
Threshold zero measures execution cost; it does not establish task accuracy.

Five clean trials alternate execution order. Optional profiler runs follow those
trials and save CPU/CUDA traces, operator summaries, and phase measurements.
Profiler timing must not replace clean timing when reporting speedup.

Report generation time, prefill time, total request time, and forward counts
separately. These are fixed-batch model measurements, not HTTP serving throughput.
Qwen produces its first token during prefill and takes 255 subsequent decode steps.
SDAR generates all 256 tokens during the measured generation interval.

At threshold zero, separate execution takes `2G` forwards for `G` output blocks.
Pairing takes `G + 1` forwards, with two blocks in each paired forward.
For 256 output tokens, B16 changes 32 calls to 17; B32 changes 16 calls to 9.
These counts are not measured speedups. Repeated denoising limits the benefit
at higher thresholds.

## Before production integration

- Measure checkpoint equivalence and speed on A100.
- Validate more prompts, output lengths, and confidence boundaries.
- Add scheduler support for allocation, mixed batches, termination, and cancellation.
- Validate production CUDA Graph capture and replay with that scheduler.
- Run task-quality and serving benchmarks before claiming an end-to-end gain.
