# Kimi-K3 DFlash optimization handoff

Status snapshot: 2026-07-29

## Branch state

- Branch: `dcw02/kimi-k3-dflash-optimization`
- Base: `origin/kimi-k3` at `40feea2700`
- HEAD: `22a27dc96a`
- The branch was rebased onto the above `kimi-k3` commit, published, and the
  runtime environment was refreshed.
- The published branch has five commits:

| Commit | Subject |
| --- | --- |
| `eca06834e4` | `minimal dflash hooks + benchmarking` |
| `9d1b390708` | `optimize block size 16` |
| `d2a59818cd` | `add peak tps` |
| `380a6bbaa4` | `block size 8 crash` |
| `22a27dc96a` | `fix mamba radix checkpoint tracking` |

In addition to this handoff, there is one intentional, uncommitted change
described below:

- exact per-request JSONL tracing in `benchmark/dflash/benchmark.py`.

`AGENTS.md` is an untracked user file and was not created or modified as part
of this work.

No unit tests were added or run, per request. Validation used the benchmark
driver and the benchmark-level KDA correctness/reference harness.

## Executive summary

This branch does four main things:

1. Adds a reproducible Kimi-K3 TP8/B300 DFlash benchmark driver, including
   decode-only TPS, peak request decode TPS, acceptance metrics, repeat runs,
   and exact request traces.
2. Minimally enables DFlash with Kimi-K3 and permits `trtllm_mha` for a DFlash
   draft checkpoint only when every draft layer is sliding attention.
3. Adds and tunes a CuTeDSL KDA verify specialization for the exact
   concurrency-1/block-16 shape.
4. Fixes two independent correctness bugs: a two-CTA GEMM lifetime race
   exposed by block 8, and DFlash Mamba radix checkpoint tracking.

The block-16 CuTe work also fixed a large systematic CuTe-versus-Triton
numerical/acceptance discrepancy. The fused CuTe kernel had skipped BF16
materialization boundaries that exist in the reference Triton composition.
Those boundaries are now explicit.

A smaller unresolved issue remains: separate fresh-server CuTe runs can
produce different greedy token streams, verify counts, and acceptance
histograms. Two fresh-server runs using the Triton KDA verify path were exactly
identical on the isolated three-request set. The CuTe drift reproduces in both
the optimized split-V path and the serial CuTe path, so the split-V
optimization itself is not the root cause.

## Commit-by-commit details

### `eca06834e4`: minimal DFlash hooks and benchmark driver

Added `benchmark/dflash/benchmark.py` with:

- target-only baseline and DFlash sweeps;
- DFlash block-size sweeps;
- HumanEval, MBPP, GSM8K, MATH-500, and MT-Bench workloads;
- a fresh server for each run/configuration;
- post-startup warmup, cache flush, measured requests, and graceful shutdown;
- overall output TPS, decode-only output TPS, aggregate acceptance length,
  equal-weight per-request acceptance length, repeat-run summaries, CSV output,
  and partial-failure reporting;
- ReplaySSM enabled by default, with a power-of-two cache length at least twice
  the DFlash block size.

Decode-only TPS is defined as:

```text
sum(max(completion_tokens - 1, 0))
-----------------------------------------------------------
duration(union([prefill_finished_time, last_decode_finish_time]))
```

The first completion token is attributed to prefill. The interval union avoids
double-counting overlapping requests at concurrency greater than one.

To make that metric available, `enable_metrics`, `prefill_finished_time`, and
`last_decode_finish_time` are preserved through the scheduler, detokenizer, and
tokenizer IPC path and returned in `/generate` metadata.

The runtime changes are intentionally narrow:

- Kimi-K3 exposes `set_dflash_layers_to_capture` as the same implementation as
  `set_dspark_layers_to_capture`.
- `--speculative-draft-attention-backend trtllm_mha` is admitted for DFlash
  only when config inspection returns exactly `num_hidden_layers` layer types
  and every entry is `sliding_attention`. Otherwise it warns and falls back to
  the normal CUDA/ROCm backend.
- The draft-worker backend allowlist includes `trtllm_mha`.

The benchmark environment was cleaned up from the old prerelease command:

- `SGLANG_K3_ATTN_RES_MODE`, `SGLANG_MOE_FUSED_GATE_RADIX`, and
  `SGLANG_RAGGED_VERIFY_MODE` are not set;
- `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` is set only for DFlash runs;
- `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` remains;
- `SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0` remains;
- the TRTLLM-Gen MoE cubin pool remains a configurable override. Passing
  `--trtllm-gen-moe-cubin-pool ""` disables the override.

`trtllm_mha` versus FA4 in this paragraph refers to the **draft model
attention backend**. It is separate from CuTeDSL versus Triton for the
**target KDA verify backend**.

### `9d1b390708`: block-16 CuTe KDA verify optimization

The existing fused CuTe KDA path handled small verify widths serially. This
commit adds a specialization for exactly one request and 16 verify tokens:

- eight CTAs per KDA head;
- each CTA owns 16 of the 128 V/output rows;
- fixed 512-thread CTAs;
- full-warp K reductions in the recurrent phase;
- a tuned 4 Q / 4 K / 7 gate / 1 V warp allocation for convolution/gating;
- one recurrent-state shared-memory stage;
- token-loop unrolling;
- one owner CTA for shared K/gate/beta rings and Q/K rollback state;
- per-CTA ownership of the V ring and V rollback state;
- invocation-local FP32 RMS partials followed by an eight-CTA cluster
  reduction for fused gated RMSNorm.

The dispatcher selects this path only for the exact dense `N=1, T=16` shape.
Other shapes keep the existing serial CuTe path or fall back.

#### BF16 materialization correctness fix

The Triton reference is a composition of separate kernels/tensors, so it has
two semantically important BF16 boundaries:

1. Q, K, and V convolution/SiLU output is materialized as BF16 before Q/K
   normalization and the recurrent update.
2. The recurrent output is materialized as BF16 before gated RMSNorm.

The fused CuTe kernel had retained FP32 values across both boundaries. That is
not merely a harmless fusion difference: it changes the recurrent trajectory
and target logits enough to create a large acceptance discrepancy.

The CuTe kernel now explicitly round-trips:

```text
post-convolution Q/K/V: FP32 -> BF16 -> FP32
pre-RMSNorm recurrent output: FP32 -> BF16 -> FP32
```

This removed the large systematic CuTe-versus-Triton acceptance bias observed
during the port. The remaining differences are much smaller and are described
under “Open CuTe run-to-run drift.”

The exact pre/post end-to-end aggregate from that early porting run was not
saved as a durable artifact. The code-level boundary, later kernel-reference
checks, and subsequent request traces are preserved; do not invent a pre-fix
number when reporting it.

The same commit adds
`benchmark/bench_linear_attention/bench_kda_mtp_verify.py`, which can:

- compare split CuTe, serial CuTe, and the repository Triton reference path;
- compare ReplaySSM raw rings and rollback convolution windows;
- compare committed states for every accepted prefix;
- compare ReplaySSM commit with non-ring per-token snapshots;
- check repeated CUDA graph replay and padded/null requests;
- microbenchmark or profile each provider.

The harness deliberately allows a one-BF16-ULP raw Q/K/V difference when the
CuTe and Triton FP32 operation order lands on opposite sides of a BF16 tie. It
checks close agreement, not full bitwise CuTe/Triton equivalence.

### `d2a59818cd`: peak request decode TPS

Added:

```text
max(
    (completion_tokens - 1)
    / (last_decode_finish_time - prefill_finished_time)
)
```

The value is reported in the per-run line, summary tables, payloads, and CSV.
Across repeated runs it is the maximum observed request, not a mean. It is an
extreme-value metric and is therefore sensitive to request length, outliers,
and the number of runs.

### `380a6bbaa4`: block-8 launch failure

Block 8 exposed an asynchronous CUDA launch failure in the generic two-CTA
CuTe BF16 GEMM. A CTA could exit while its cluster peer could still address its
shared memory.

The fix adds a final cluster arrive/wait for the two-CTA variant, keeping both
CTAs resident until neither can target its peer’s shared memory. Although the
commit title names block 8, the root fix is in the shared GEMM and is not
DFlash/KDA-specific.

The benchmark also sets `SGLANG_PYSPY_DUMP_BEFORE_CRASH=0`, avoiding the noisy
and permission-denied py-spy attempts seen after the CUDA failure.

### `22a27dc96a`: DFlash Mamba radix checkpoint tracking

DFlash did not call `prepare_mamba_track_for_verify()` before target verify and
used a potentially stale/reservation-inflated sequence length to decide which
intermediate Mamba state crossed a checkpoint interval.

The fix:

- prepares Mamba tracking immediately before target verify;
- computes the accepted post-verify sequence lengths before state commit;
- passes explicit pre- and post-verify sequence lengths;
- chooses the interval-crossing step from the actual accepted length.

Without this, a speculative verify crossing `mamba_track_interval` could miss
or store the wrong Mamba checkpoint. The bad checkpoint becomes visible when a
radix/prefix-cache resume later reuses it, at which point the request can
continue from the wrong recurrent state.

This is primarily a correctness issue for hybrid-Mamba radix extra-buffer
tracking and later prefix reuse. No controlled performance delta was recorded;
the added bookkeeping is small relative to target verify. DFlash with
`extra_buffer_lazy` remains explicitly unsupported pending lifecycle
validation.

## Open CuTe run-to-run drift

### What is established

The current Triton KDA verify path is the best numerical reference available in
this repository:

- it is the unfused composition of the existing Triton convolution,
  recurrent, and gated-norm kernels;
- ReplaySSM’s exact-fold commit intentionally follows the Triton recurrent
  operation order;
- it exposes the BF16 tensor boundaries that the fused kernel must preserve;
- on the isolated three-request test, two fresh servers produced identical
  output token IDs, completion lengths, verify counts, correct/proposed draft
  counts, and complete acceptance histograms.

Higher acceptance is useful evidence but is not itself a correctness
criterion. One serial CuTe diagnostic run had higher aggregate acceptance than
Triton while still producing different tokens and histograms. Reference status
comes from semantics and repeatability, not from maximizing acceptance.

The aggregate acceptance difference and the difference in mean per-request
acceptance between CuTe and Triton cannot be evaluated properly until CuTe is
exactly stable across fresh-server runs. The current between-provider
comparison is confounded by CuTe's within-provider run-to-run variation. Do not
claim that either backend has higher or lower acceptance from these runs.

### Initial draft-attention observation

The investigation started with one HumanEval run per DFlash **draft attention**
backend:

| Draft backend | Overall output TPS | Aggregate acceptance | Mean request acceptance |
| --- | ---: | ---: | ---: |
| `trtllm_mha` | 357.33 | 7.663 | 8.211 |
| FA4 | 361.72 | 7.904 | 8.279 |

These single runs were not enough to distinguish numerical drift from
fresh-server variation. They motivated repeat runs and exact request tracing.
They must not be quoted as the CuTe-versus-Triton target KDA comparison.

### Full HumanEval fresh-server comparison

Default split CuTe, 164 HumanEval requests, two fresh servers:

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Completion tokens | 170,408 | 163,877 |
| Verify calls | 22,001 | 20,855 |
| Aggregate acceptance | 7.745466 | 7.857924 |
| Mean request acceptance | 8.264764 | 8.283244 |

- Prompts were identical.
- All 164 output token streams differed.
- 161/164 verify counts differed.
- All 164 acceptance histograms differed.
- Median first output-token divergence was token 28; range was 0–537.

The aggregate acceptance difference looks modest, but a tiny numerical change
near an argmax boundary can alter one greedy token and then amplify through the
rest of a long continuation.

### Three-request provider isolation

All rows below used HumanEval/0–2, block 16, ReplaySSM, concurrency 1, and a
fresh server for each run.

Triton target KDA verify:

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Completion tokens | 1,450 | 1,450 |
| Verify calls | 182 | 182 |
| Aggregate acceptance | 7.967033 | 7.967033 |
| Mean request acceptance | 7.974949 | 7.974949 |
| Decode-only TPS | 438.143 | 438.806 |

The exact output IDs, output lengths, verify counts, correct/proposed counts,
and full histograms were identical.

Split CuTe during the subsequently reverted PDL ordering experiment:

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Completion tokens | 1,536 | 1,536 |
| Verify calls | 194 | 213 |
| Aggregate acceptance | 7.917526 | 7.211268 |
| Mean request acceptance | 8.061883 | 7.273088 |
| Decode-only TPS | 501.957 | 458.597 |

All three output streams and histograms differed. The first-divergence
positions were 30, 103, and 2.

Serial CuTe during the same reverted experiment:

| Metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Completion tokens | 1,468 | 1,536 |
| Verify calls | 176 | 197 |
| Aggregate acceptance | 8.340909 | 7.796954 |
| Mean request acceptance | 8.364193 | 7.813757 |
| Decode-only TPS | 516.417 | 484.113 |

All three output streams and histograms also differed. This rules out the
eight-CTA split-V reduction as the primary cause.

A 16-output-token control produced identical output streams, while one request
still shifted one accepted-token histogram bucket. Acceptance can therefore
drift even before it changes the externally visible token stream.

The identical-input kernel harness is bitwise stable across 100 CUDA graph
replays. The observed drift is associated with fresh-server/full-model input
or state trajectories, not random execution of a fixed captured kernel input.

Fixed TP all-reduce ordering, `--enable-deterministic-inference`, and a coarse
WAR-ordering control did not remove the CuTe drift. A three-request target-only
baseline also differed across fresh servers, so the wider K3/TP8/B300 target
stack is not globally bitwise deterministic. The exact two-run Triton result
is nevertheless a strong path-specific control, not a proof for every prompt.

### Kernel-level numerical evidence

The current correctness harness passed for seeds 0, 17, and 42:

```text
max split/serial relative output error: 2.47e-05
max CuTe/Triton relative output error:  1.60e-04
max committed-state relative error:     7.08e-10
max ReplaySSM snapshot relative error:  2.28e-07
```

Across broader one-off seeds, typically only 0–7 out of 24,576 BF16 output
values per layer differed between split CuTe and Triton, usually by one BF16
ULP. The layouts, ring addresses, BF16 boundaries, and formulas appear
semantically aligned; the remaining differences are rare arithmetic ties.

### Most likely remaining sources

1. CuTe reassociates the recurrent update. It precomputes `decay * k` and
   `beta * k`, then uses packed FMAs. Triton executes:

   ```text
   h *= decay
   v -= sum(h * k)
   v *= beta
   h += k * v
   output = sum(h * q)
   ```

   These are algebraically equivalent but not FP32-bitwise equivalent.

2. ReplaySSM commits with the Triton operation order. CuTe verify can therefore
   produce logits from a slightly different transient trajectory than the
   trajectory installed by commit. The measured state error is tiny, but it is
   not exactly zero.

3. CuTe uses explicit approximate exp/reciprocal/rsqrt operations and a
   different reduction tree for Q/K normalization, gate/beta formation, and
   output norm.

4. CuTe and Triton convolution use different FP32 operation order. The BF16
   boundary limits this mostly to rare one-ULP ties.

5. Split-V gated RMSNorm uses eight CTA partial sums rather than Triton’s
   single-program reduction. This is not the main source because serial CuTe
   also drifts, but it is another obstacle to full bitwise parity.

### Recommended next step

Do not optimize against aggregate acceptance. Capture one real KDA layer input
at the first request/token that diverges and compare stages in this order:

1. raw convolution Q/K/V and ring gate/beta;
2. every accepted-prefix recurrent state;
3. pre-output-norm BF16 recurrence output;
4. final gated RMSNorm output.

The first actual CuTe kernel change should rewrite the recurrent phase in the
same ordered sequence as Triton and use division-form Q/K normalization. Tighten
the ReplaySSM snapshot comparison to exact equality for every prefix 1–16.
After that, isolate convolution and output norm independently. Whole-output
bitwise equality will require matching operation and reduction order at every
stage; it is not a one-line fast-math change.

Only after two or more fresh-server CuTe runs are exact in output tokens,
verify counts, and acceptance histograms should aggregate acceptance and mean
per-request acceptance be compared with Triton.

If correctness must take priority immediately, the narrow safe fallback is in
`KDAAttnBackend._can_run_dspark_cutedsl_mtp()`: bypass CuTe only for DFlash,
16-token, ReplaySSM verify. The `nv_cutedsl` dispatcher already installs the
Triton verify kernel as its fallback, and the normal external output norm is
used when CuTe does not consume it. This fallback was investigated but has not
been implemented because it gives up the block-16 CuTe performance work.

## Current uncommitted changes

### Exact request tracing

`benchmark/dflash/benchmark.py` adds:

```text
--request-trace-dir DIR
```

It writes one JSONL file per run with:

- stable sample/run identifiers;
- prompt-token hash;
- exact output token IDs and hash;
- completion length;
- verify count;
- correct/proposed draft counts;
- complete correct-drafts histogram;
- decode interval/TPS;
- finish reason.

Writes are atomic and records are emitted in measured sample order even though
requests may complete out of order. This tracing produced the evidence above.

## Reverted diagnostics

The PDL ordering experiment moved mutable Q/K convolution-state loads to after
`cute.arch.griddepcontrol_wait()`.

It did not improve fresh-server exactness in either split or serial CuTe, and
no controlled performance improvement was established. It was therefore
reverted so the handoff starts from the published kernel state. The two
three-request CuTe tables above were collected during this experiment and are
retained only as evidence that neither PDL load placement nor split-V explains
the drift.

## Reproduction and validation

Typical end-to-end run:

```bash
uv run python benchmark/dflash/benchmark.py \
  --workloads humaneval \
  --concurrencies 1 \
  --skip-baseline \
  --dflash-block-sizes 16 \
  --dflash-draft-model /tmp/dflash/draft-epoch-10
```

Add exact tracing with the uncommitted benchmark change:

```bash
uv run python benchmark/dflash/benchmark.py \
  --workloads humaneval \
  --concurrencies 1 \
  --skip-baseline \
  --dflash-block-sizes 16 \
  --dflash-draft-model /tmp/dflash/draft-epoch-10 \
  --num-samples 3 \
  --runs-per-config 2 \
  --max-new-tokens 512 \
  --min-warmup-generation-turns 2 \
  --request-trace-dir /tmp/k3-dflash-trace
```

The benchmark currently does not expose the **target** linear verify backend as
a pass-through flag. The Triton control used
`--linear-attn-verify-backend triton` through a temporary local server-argument
override that was subsequently reverted. This is distinct from:

```text
--speculative-draft-attention-backend trtllm_mha|fa4
```

Kernel/reference check:

```bash
uv run python benchmark/bench_linear_attention/bench_kda_mtp_verify.py \
  --mode check \
  --batch-size 1 \
  --heads 12 \
  --width 16 \
  --ring-len 32 \
  --check-seeds 0 17 42
```

Static validation run for the uncommitted changes:

```bash
git diff --check
uv run python -m py_compile benchmark/dflash/benchmark.py
```

Both passed.

The CuTe optimization was also temporarily bypassed to collect the Triton
control and temporarily forced to its serial mode to test the split-V
hypothesis. Both diagnostic edits were reverted; the committed dispatch remains
split-V for dense `N=1, T=16`.

## Local diagnostic artifacts

These paths are under `/tmp` and are not durable:

- full two-run CuTe comparison:
  `/tmp/k3-dflash-accept-default-20260729`;
- Triton two-run control:
  `/tmp/k3-dflash-triton-verify-20260729`;
- split CuTe during the reverted PDL ordering experiment:
  `/tmp/k3-dflash-cute-pdl-fix-20260729`;
- serial CuTe during the reverted PDL ordering experiment:
  `/tmp/k3-dflash-cute-serial-pdl-fix-20260729`;
- fixed-order all-reduce control:
  `/tmp/k3-dflash-fixed-order-ar-20260729`;
- deterministic-inference control:
  `/tmp/k3-dflash-deterministic-20260729`;
- coarse WAR-ordering control:
  `/tmp/k3-dflash-coarse-war-20260729`;
- 16-token output control:
  `/tmp/k3-dflash-prefix16-default-20260729`;
- target-only fresh-server control:
  `/tmp/k3-baseline-greedy-20260729`;
- original block-8 CUDA coredump:
  `/tmp/dflash_block8_coredumps`.

Copy any needed JSONL files out of `/tmp` before recycling the host.

## Intentionally not present

- `--disable-hybrid-swa-memory` was briefly considered and then explicitly
  reverted.
- No old prerelease K3 attention-residual, fused-gate-radix, or ragged-verify
  environment override is required by this benchmark.
- No broad CuTe disable/fallback has been landed.
- No claim is made that higher acceptance alone identifies the correct kernel.
