# MoE LoRA tile tuning

`tune_plans.py` searches bounded LoRA launch tiles for a resident model shape
and GPU, retaining shipped plan families and base-GEMM configurations. The result
is the best validated candidate in that space, not a globally optimal plan,
production qualification or model TPS.

## Inputs

Create a JSON list of explicit local cases, for example:

```json
[{
  "name": "decode-mixed",
  "hidden_size": 2048,
  "intermediate_size": 512,
  "num_local_experts": 256,
  "tokens": 16,
  "rank": 32,
  "top_k": 8,
  "slots": 2,
  "quant": "bf16",
  "vendor": "cutedsl",
  "layout": "per_expert",
  "phase": "decode",
  "mode": "eager",
  "routing": "balanced",
  "traffic": "mixed"
}]
```

Run from this checkout with its installed GPU dependencies:

```bash
python benchmark/kernels/lora_moe/tune_plans.py \
  --cases cases.json --out /tmp/my-moe-study \
  --repeats 6 --warmup 5 --iterations 20 --max-candidates 32
```

The output directory must not exist. Optional
`--model-config /local/config.json` supplies conventional routed-MoE geometry
instead of the three shape fields. It requires `hidden_size`,
`moe_intermediate_size`, a routed expert count and explicit SiLU activation;
inspect the model to confirm gated SiLU. Nothing is downloaded.
`--tp-size` means total TP including EP; MoE DP is fixed at one.
Local intermediate width is `I / (TP / EP)`, resident experts `E / EP`,
with exact divisibility required. Explicit local geometry is preferable for
padded, fused-shared-expert or model-specific layouts. Contradictory inferred
and explicit dimensions fail. TP/EP are identity metadata, not distributed work.

Fixtures support gated SiLU, BF16/block-128 FP8, SM90/SM100, CuTeDSL/Triton, and
per-expert/shared-outer LoRA. Rank is the physical padded rank: a multiple of
eight, at most 256. FP8 H/I must be 128-aligned. Routing is balanced/skewed;
traffic active/mixed/base_only. Every routed expert must be resident.
NVFP4, other activations, distributed dispatch/collectives and serving integration
are unsupported. Failures are recorded, not converted to another vendor.
External plan/base configuration override environments are rejected.

## Measurement and selection

The fixture invokes production `MoeLoraRunner.run`: routing, provider
preparation/base GEMMs, LoRA A/B, activation and finalization. A local subclass
changes only output allocation to ordinary `torch.empty`, because the serving
allocator requires a TP group. No production method is monkeypatched and no
backend environment is changed. Serving symmetric allocation, communication,
scheduler and model execution are excluded.

Eager timing is synchronized whole-call wall time. Graph timing warms the actual
capture stream and measures real replay, excluding graph construction and
capture-only host work. These are separate case identities, not interchangeable
latencies or sums of stage medians.

Before timing, incumbent and candidates are checked against independent FP32
gated-MoE algebra with exact effective weights. Fixed gates: finite output,
relative L2 <= 0.02 BF16 / 0.06 FP8, and elementwise atol=0.018/rtol=0.06.
Graph checks overwrite captured output with NaNs and verify replay replaces it.
These fixture gates are not model-level numerical gates.

One-axis candidates vary A/B block width, warps or stages; routing, split mode,
plan family and provider remain fixed. Paired repeats alternate AB/BA against
the incumbent. Gain is `baseline_us / candidate_us - 1`. WIN/LOSS require
unanimous signs and median magnitude >= max(2%, baseline `(max-min)/median`
spread). Otherwise a negative tail beyond 2% is INCONCLUSIVE; the rest is TIE.
A search WIN must WIN again on independent inputs and weights. Validation is
not pooled with search; a failed shortlist is not replaced until something passes.

## Results and scope

`study.json` records cases/budget, `trials.jsonl` retains all trials/failures,
and `results.json` reports decisions. No WIN retains the incumbent. Incumbent,
all-candidate or invalid-validation failure fails the case. Fatal CUDA errors
stop the study. Evidence includes source hashes, loaded-checkout checks, device
UUID, runtime versions, actual provider and resolved base configuration.

A winner contains an exact-study descriptor, **not** a production
`*.plans.json` override. Do not install it as one. Production rows cannot
restrict exact model/I/rank/token/routing identity: copying a measured tile
would affect unmeasured regions. Promotion requires separate declared coverage;
this tool neither widens domains nor writes packaged tables.

Base-GEMM optimization is separate: see
`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` for Triton.
The adjacent `tune_down_moe.py` is retained to screen a distinct down-GEMM
configuration while holding gate/up fixed. Both are screening tools: their
output still needs numerical and full-pipeline validation. This tuner does
not inject custom CuTeDSL base tables or jointly optimize base and LoRA kernels.

## Retired experiments

The old seed/variant writers, partial CuTeDSL base-stage sweep and FP8 profiler
are available in Git history, not maintained tuning entrypoints. Blind domain
widening, outdated tile-schema emission and stage-profiler ranking are not part
of this workflow. Production table format/selection documentation remains in
`python/sglang/srt/lora/moe/configs/README.md`.

CPU contracts: `test/registered/unit/lora/test_moe_lora_tuning.py`.
CPU tests alone do not establish GPU compilation, numerics or performance.
