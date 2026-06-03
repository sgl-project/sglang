# Task (tom): Kimi-K2.5 down-proj LoRA-B expand-add — shapes + BLOCK_SIZE_N tuning, e2e-validated

> **This file is the single source of truth for the PR** (jybsuper/sglang **#12**,
> `fzyzcjy:tom/lora-moe-expand-down-bench` → `nvfp4-lora`). The PR description is intentionally
> empty — read this journal. The PR is **stacked on #11**: until #11 merges, the PR diff also
> shows #11's `bench_expand_add_down.py` + UT + `JOURNAL.md`; the net-new commits are the three
> listed under "What changed".

Follow-up to PR #11 (`bench_expand_add_down.py`), which scoped the down-proj expand-add to
**qwen3.5-35b** (64 local experts, N=2048). This:
1. Adds the **Kimi-K2.5** shape (`--model kimi-k25`).
2. Makes **BLOCK_SIZE_N** tunable and sweeps it.
3. **Validates the micro-bench against an e2e decode trace** (the headline check: a tuning
   number only counts if the micro-bench reproduces what production actually runs).

GPU: 1× NVIDIA GB200 (leira / Crusoe, pod `sglang-lora-tom` on node `np-67167b3f-16`), image
`lmsysorg/sglang:dev-cu13`, branch `tom/lora-moe-expand-down-bench` (on PR #11 head
`a4a46f3de`). No model weights — pure triton kernel testbed.

## What changed

1. `python/.../lora/trtllm_moe/specialized_expand.py` — `_invoke_moe_lora_expand_add` gains an
   opt-in `force_block_size_n: int | None = None`. Default `None` → byte-identical to before
   (`128 if N % 128 == 0 else config["BLOCK_SIZE_N"]`). When set, it overrides that so a
   tuner/bench can explore BLOCK_SIZE_N. Down-proj is non-gated (`gated_a_half = 0`), so any
   divisor of N is valid; the gated gate_up split still asserts `N/2 % block_size_n == 0`.
2. `benchmark/kernels/lora_moe_expand/bench_expand_add_down.py`
   - `--model {qwen35, kimi-k25}` preset (per-flag overrides win).
   - `--mode sweep` now also sweeps `BLOCK_SIZE_N ∈ {64,128,256,512}` (divisors of N) via
     `force_block_size_n`; `--block-n` forces it in single-config modes.
   - **`--config {production, manual}` (default `production`)**: by default `--mode bench`/
     `profile` reproduce the config production launches — `production_config()` queries the same
     `try_get_optimal_moe_config` path `_get_stage_config` uses (BLOCK_SIZE_N left to the
     launcher). So **the default run matches the e2e kernel**, not a hand-picked config.
     `--config manual` uses the explicit `--block-*` / `--group-m` / `--num-warps` flags (A/B,
     tuning). Verified default numbers: **kimi-k25 → 27.3 µs** (block_m=16, block_n=128, warps=4;
     ≈ e2e 25.1 µs); **qwen35 → 6.49 µs** (same config).

## Are PR #11's shapes wrong for Kimi? — Yes, they were qwen's

Kimi-K2.5 down-proj LoRA-B shape, from the adapter + config (local note
`kimi-k25-shapes-roofline.md` §3 MoE + §4 LoRA):

| dim | PR #11 (qwen3.5-35b) | Kimi-K2.5 |
|-----|----------------------|-----------|
| routed experts | 64 (tp4/ep4 local) | **384** (TP8, no-EP — all experts present per rank) |
| N = down output hidden | 2048 | **7168** (= hidden H) |
| rank | 16 | 16 |
| top_k | 8 | 8 |

Down-proj per-expert is `[2048×7168]`; the LoRA-B (expand) weight is `[r=16 × out=7168]`, so the
expand-add kernel sees **N=7168, R=16, 384 experts, top_k=8**. PR #11's N=2048 / 64 experts are
qwen's numbers, not Kimi's.

## The production config — and reproducing the e2e trace

**This is the part that makes the numbers count.** An e2e decode trace (graph step
`step[DECODE bs=64]`) shows `_moe_lora_expand_add_kernel` at **25.1 µs**. The micro-bench must
reproduce that with the *same config production picks*, otherwise tuning deltas are meaningless.

The production config comes from `_get_stage_config` (`virtual_experts.py`) →
`try_get_optimal_moe_config(lora_b_virtual.shape, …, stage_top_k=1, dtype=bf16, M=bs)`. Probed
in the pod for the Kimi expand shape `[E=384, N=7168, R=16]`:

```python
# (server args must be set, else try_get_optimal_moe_config raises and _get_stage_config
#  falls back to BLOCK_SIZE_M=64 — see the Correction note below)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_dtype_str, try_get_optimal_moe_config)
import functools, torch
w = (384, 7168, 16); dt = get_config_dtype_str(dtype=torch.bfloat16)
f = functools.partial(try_get_optimal_moe_config, w, w, 1, dt)
print(f(64))   # M=bs=64
# -> {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}  (M=16/32/64 identical)
```

There is **no tuned JSON** for this shape (`E=384,N=16,...GB200.json` not found), so it returns
the library default `BLOCK_SIZE_M=16, GROUP_SIZE_M=1` (no `num_warps` key → launcher uses
`config.get("num_warps", 4) = 4`). The launcher then forces `BLOCK_SIZE_N=128` (7168 % 128 == 0).

**So the real production config = `block_m=16, block_n=128, group_m=1, num_warps=4`.** Micro-bench:

```
block_m=16 block_n=128 group_m=1 warps=4  ->  27.13 µs   (production-equivalent)
block_m=16 block_n=128 group_m=1 warps=2  ->  22.57 µs
```

The e2e **25.1 µs** sits between these and ~8% under the warps=4 number — well within
micro-bench-vs-e2e noise (single trace sample, surrounding-kernel state, warmup). **The
micro-bench reproduces the e2e production kernel.** ✓

### Correction (supersedes an earlier draft of this file)
An earlier draft claimed the production default was `block_m=64` → 114 µs → a "5.68× win". That
was **wrong**: `block_m=64` is the `_get_stage_config` *fallback* used only when
`try_get_optimal_moe_config` raises (e.g. no server args). In real serving the server args are
set and it returns `block_m=16`, giving ~25–27 µs — matching the e2e trace. The corrected win
from tuning is below.

This also applies to #11's qwen claim: with `--config production`, qwen35 resolves to
`block_m=16` too (no tuned JSON either) → **6.49 µs**, i.e. qwen production already runs the
`block_m=16` #11 called "best", not the `block_m=64` it called the default. **For both models the
real untuned lever is `BLOCK_SIZE_N` (forced 128), not `block_m`.**

## Results (1× GB200, bs=64, r=16, amortized device time)

### Correctness (`--mode correctness --model kimi-k25`) — 6/6 PASS
fp32 ref vs per-expert bf16 atomic-add, tol 5e-2 abs; block_m {16,32,64} × bs {16,64}:
`max_abs_err` 2.6e-3–3.2e-3 (rel 6.5e-3–8.3e-3).

### Sweep (`--mode sweep --model kimi-k25`) — block_m × block_n × group_m × num_warps
**BEST: 20.12 µs `block_m=16 block_n=512 group_m=1 warps=4`.**

Min µs per (block_m, block_n) over group_m/num_warps:

| block_m \ block_n | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| **16** | 32.4 | 22.5 | 21.4 | **20.1** |
| 32 | 38.9 | 31.8 | 30.5 | 33.0 |
| 64 | 57.5 | 65.6 | 64.6 | 88.8 |

- `block_m=16` dominates (production already uses it — consistent with the e2e number).
- At block_m=16, **larger block_n helps**: 64→512 takes 32.4→20.1 µs. The production-forced
  `block_n=128` is *not* optimal here.

### A/B vs the real production config

| config | µs (bench) | note |
|---|---|---|
| **production** `block_m=16 block_n=128(forced) warps=4` | **27.1** | ≈ **25.1 µs e2e** |
| **BEST** `block_m=16 block_n=512 warps=4` | **20.1** | force_block_size_n=512 |

→ Tuning **BLOCK_SIZE_N 128 → 512** at the production `block_m=16` gives **~1.35× vs the
micro-bench production (27.1→20.1 µs)**, or **~1.25× vs the e2e 25.1 µs**. Not the 5.68× of the
earlier (wrong) draft — the real, e2e-anchored win is ~25%.

### Finding (for a follow-up — NOT a kernel/config change here)
The down-proj expand at the Kimi shape (bs=64, r=16, N=7168, 384 experts) is **~25% faster with
`BLOCK_SIZE_N=512`** than the launcher's forced 128. Shipping it means letting the down-proj
expand pick block_n=512 (a generated/tuned config or a down-proj branch in the launcher), then
validating with the full Qwen+Kimi regression + perf-benchmark flow. This PR only adds the
testbed + the `force_block_size_n` knob.

## Reproduce

```bash
cd /root/sglang   # pod sglang-lora-tom, branch tom/lora-moe-expand-down-bench
B=benchmark/kernels/lora_moe_expand/bench_expand_add_down.py
python3 $B --mode correctness --model kimi-k25                       # 6/6 PASS
python3 $B --mode sweep       --model kimi-k25                       # BEST bm16 bn512 ~20us
python3 $B --mode bench       --model kimi-k25                       # default=production ~27us (=25.1 e2e)
python3 $B --mode bench       --model kimi-k25 --config manual --block-m 16 --block-n 512   # BEST ~20us
# production config probe: see the python snippet in "The production config" section above.
```
