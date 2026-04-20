# Dynamic-Threshold Heter Dispatch Policy (`dyna_thresh`)

**Date:** 2026-04-20
**Status:** Proposed
**Owner:** huanchen

## Summary

Introduce `dyna_thresh` — a heter-dispatch policy whose `expert_batch` threshold
is a **step function of scheduler concurrency** (`ForwardBatch.batch_size`)
rather than a scalar constant. An offline analyzer under
`expert_precision_assignment/policy/dyna_thresh/` consumes the existing
`(mc × thr)` serving sweep and emits a per-band winning threshold. The runtime
policy does one bisect per forward over a small sorted array, then gates on
`n >= thr` exactly like the existing `expert_batch` policy.

This is the closed-loop cousin of the static `expert_batch` policy defined in
`2026-04-15-batch-size-gated-policy-design.md`: same gating mechanism, same
CUDA-graph guarantees, but the threshold now tracks live concurrency so per-
forward latency stays within the window the profile says is achievable at that
load.

## Motivation

The `expert_batch` policy (shipped) uses a single scalar threshold (default 128).
The profile sweep (`profile/gen_dyna_variants.py`) already measures performance
across `mc ∈ {8, 16, 32, 64, 128, 256}` × `thr ∈ {32, 64, 128, 256, 512}`, and
the winning threshold is not the same across concurrency levels — a threshold
that is good at `mc=8` is typically wrong at `mc=256`. Keeping a single constant
leaves per-forward time on the table at either end.

`dyna_thresh` closes that gap by letting the runtime pick the threshold
appropriate for the current concurrency band, without adding measurable forward
overhead.

Why scheduler-side concurrency (not token count `n`) drives the choice:
the profiling pipeline's primary axis is `mc` — we already have measured data
organized by concurrent-request count. Using the same axis at runtime means
the mapping we emit is directly anchored to measurements, not to a proxy.

## Design

### Components

Two new pieces, one existing runtime extension.

1. **Offline analyzer** — `expert_precision_assignment/policy/dyna_thresh/`
   - `pick_thresholds.py` — reads `profile/results/*.jsonl`, groups records by
     `mc`, picks the winning `thr` per band under a configurable objective,
     writes `dyna_thresh_config.json`.
   - `test_configs.py` — validates analyzer output shape, band coverage, and
     monotonic ordering.
   - `vram_estimator.py` — shared with `heter_assign/` (imported, not copied).

2. **Runtime dispatch policy** — `python/sglang/srt/layers/moe/heter_policy.py`
   - Add `DynaThreshHeterDispatch`, registered in `_POLICY_REGISTRY` under the
     key `"dyna_thresh"`.
   - Reuses `ExpertBatchGatedHeterDispatch`'s gating mechanism; only the
     threshold source differs.

3. **Config schema** — unchanged wrapper, new `policy` string and
   `policy_params`.

### Runtime policy

```python
class DynaThreshHeterDispatch(HeterDispatchPolicy):
    """Gate hot/cold by current batch token count, where the threshold is
    a step function of scheduler concurrency.

    bands: list of (max_concurrency, threshold) pairs, sorted ascending by
    max_concurrency. For current scheduler batch_size c:
        thr = bands[bisect_left(mc_keys, c)].threshold
    Then gate on n >= thr, same as ExpertBatchGatedHeterDispatch.
    """

    def __init__(self, num_experts, group_size_ratios,
                 bands, device=None,
                 int4_only_mask=None, int4_group_idx=0,
                 bf16_group_idx=None):
        super().__init__(...)
        # bands = [[mc_bucket, thr], ...] sorted by mc_bucket asc.
        self._mc_keys = [b[0] for b in bands]
        self._thrs = [b[1] for b in bands]
        self._int4_group_idx = int4_group_idx
        self._bf16_group_idx = (
            bf16_group_idx if bf16_group_idx is not None
            else (1 - int4_group_idx)
        )
        self._has_int4_only = (
            int4_only_mask is not None and bool(int4_only_mask.any().item())
        )

    def _pick_threshold(self, c):
        # Bisect: first band whose mc_bucket >= c. If c exceeds all buckets,
        # clamp to the last (highest-concurrency) band.
        i = bisect.bisect_left(self._mc_keys, c)
        if i >= len(self._thrs):
            i = len(self._thrs) - 1
        return self._thrs[i]

    def _assign(self, token_selected_experts, token_final_scales,
                forward_batch=None):
        if token_selected_experts is None:
            self._expert_to_group_buf.fill_(self._int4_group_idx)
            return self._expert_to_group_buf

        n = token_selected_experts.shape[0]
        c = forward_batch.batch_size if forward_batch is not None else n
        thr = self._pick_threshold(c)
        target = (
            self._bf16_group_idx if n >= thr
            else self._int4_group_idx
        )
        self._expert_to_group_buf.fill_(target)
        return self._expert_to_group_buf

    def should_skip_group(self, group_idx, num_tokens, forward_batch=None):
        c = forward_batch.batch_size if forward_batch is not None else num_tokens
        thr = self._pick_threshold(c)
        if num_tokens >= thr:
            if group_idx == self._int4_group_idx:
                return not self._has_int4_only
            return False
        else:
            return group_idx == self._bf16_group_idx
```

### Forward-batch plumbing

`ForwardBatch.batch_size` already carries the per-forward request count
(`forward_batch_info.py:286`). No new plumbing is required; the MoE layer's
`forward` receives `forward_batch` today and will pass it through to
`policy._assign` and `policy.should_skip_group` as a new optional kwarg.

Existing policies (`expert_batch`, `expert_load`, `total_weight`, `confidence`,
`random`) gain `forward_batch=None` in their signatures and ignore it. Only
`DynaThreshHeterDispatch` consumes it.

### CUDA-graph safety

Same argument as `expert_batch`: all decisions are Python-side on scalars
(`n = tensor.shape[0]`, `c = forward_batch.batch_size`, both host ints).
No GPU→CPU sync of tensor values. Each captured graph observes one `(n, c)`
pair and bakes in one branch. Eager and dynamic-shape paths work unchanged.

### Config schema

```json
{
  "policy": "dyna_thresh",
  "policy_params": {
    "bands": [
      [8,   512],
      [16,  256],
      [32,  128],
      [64,   64],
      [128,  64],
      [256,  32]
    ]
  },
  "groups": [
    {"name": "cold", "num_bits": 4, "group_size": 128,
     "checkpoint": "/path/to/gptq"},
    {"name": "hot",  "num_bits": 16}
  ],
  "int4_only_experts_file": "/path/to/int4_only_experts.json"
}
```

- `bands` is required, non-empty, sorted ascending by the first element
  (`max_concurrency`), and all thresholds must be positive ints.
- `int4_only_experts_file` continues to force int4-only experts into the INT4
  group regardless of the gate.

### Offline analyzer

`pick_thresholds.py` CLI:

```
python pick_thresholds.py \
  --sweep_jsonl expert_precision_assignment/profile/results/*.jsonl \
  --objective {mfs,itl,throughput,pareto} \
  --out_file configs/dyna_thresh_config.json \
  [--heter_base configs/mc128/heter_config.json]
```

- Reads sweep rows `{mc, thr, variant_name, metrics: {...}}`.
- Keeps only `variant_name` matching `thr{N}` (skips the `hot{N}` ratio
  variants — those exercise a different policy).
- Groups by `mc`; for each group, picks the `thr` that minimizes/maximizes the
  selected objective.
- Emits a config with `policy="dyna_thresh"`, band table, and the groups +
  `int4_only_experts_file` carried over from `--heter_base` (so the output is a
  drop-in replacement for `heter_config.json`).

The four `--objective` choices each map to a scalar over the metrics JSON:

| Flag | Metric | Direction |
|---|---|---|
| `mfs` | per-forward time (`model_forward_s`) | minimize |
| `itl` | median inter-token latency (`median_itl_ms`) | minimize |
| `throughput` | output tokens/sec (`output_throughput`) | maximize |
| `pareto` | min `mfs` s.t. `itl` within user-supplied ceiling | minimize |

The concrete metric keys come from the sweep JSONL — the analyzer will be
written against whatever `collect_results.py` produces today (no schema
invention). If `pareto` is selected without a `--itl_ceiling_ms`, the analyzer
errors out.

The **choice of default objective is deferred** until the profile is inspected;
the analyzer ships with all four so the decision is a flag change, not a code
change.

### Interaction with existing components

| Component | Behavior |
|---|---|
| `_int4_only_mask` | Still forces int4-only experts into `_int4_group_idx`. |
| `_bf16_id_remap` | Unchanged — only consumed when the BF16 kernel runs. |
| `expert_batch` policy | Unchanged. `dyna_thresh` is a sibling, not a replacement. |
| Sweep scripts | Unchanged. `dyna_thresh` consumes their existing output. |
| `HeterFusedMoE.forward` | One-line change: pass `forward_batch` into policy calls. |

## Testing

New unit tests in `test/test_heter_moe/unittest/test_dispatch_policy.py`:

1. `test_dyna_thresh_band_lookup_exact` — `c` exactly matches a band boundary →
   picks that band's threshold.
2. `test_dyna_thresh_band_lookup_between` — `c` between two bands → picks the
   upper neighbor's threshold (bisect_left semantics).
3. `test_dyna_thresh_band_lookup_clamp_high` — `c` > all bands → picks last band.
4. `test_dyna_thresh_band_lookup_clamp_low` — `c` < first band → picks first band.
5. `test_dyna_thresh_int4_only_mask_respected` — even when the gate is BF16,
   experts in `int4_only_mask` end up in INT4.
6. `test_dyna_thresh_should_skip_group` — truth table across `c` × `n` ×
   `_has_int4_only` matches the gated-policy contract.
7. `test_dyna_thresh_invalid_config` — unsorted bands / empty bands / non-int
   thresholds → raises at policy init.

New end-to-end correctness check in
`test/test_heter_moe/unittest/test_correctness.py`:

8. `test_dyna_thresh_matches_static` — build a `dyna_thresh` config with a
   single band, run forward, verify outputs match the equivalent static
   `expert_batch` config. This pins dynamic == static when the mapping is
   degenerate.

CUDA-graph regression in `test_efficiency.py`:

9. `test_dyna_thresh_cuda_graph` — capture graphs at two concurrency levels
   that resolve to different thresholds, and verify both replay successfully.

Analyzer smoke test (`expert_precision_assignment/policy/dyna_thresh/test_configs.py`):

10. Synthetic sweep JSONL → analyzer produces a valid config (bands sorted,
    thresholds positive, band count equals number of distinct `mc` values in
    the sweep).

## Out of scope

- **Closed-loop / MFS-feedback controller.** The policy is open-loop: the
  band table is fixed at config load. A feedback controller that measures MFS
  and drifts the threshold is future work.
- **Interpolation between bands.** Step function only. Linear interpolation
  would mask measurement noise and couple unrelated bands.
- **Multi-GPU concurrency aggregation.** Assumes single-GPU, matching the rest
  of the sensitivity pipeline. Under TP, every rank sees the same
  `ForwardBatch.batch_size`, so the policy is consistent across ranks without
  additional collectives.
- **New sweep axes.** `dyna_thresh` reuses the existing sweep — no new
  benchmarks added as part of this work.

## Files touched

New:
- `expert_precision_assignment/policy/dyna_thresh/pick_thresholds.py`
- `expert_precision_assignment/policy/dyna_thresh/test_configs.py`
- (optional symlink or shared import) `expert_precision_assignment/policy/dyna_thresh/vram_estimator.py`

Modified:
- `python/sglang/srt/layers/moe/heter_policy.py`
  - Add `DynaThreshHeterDispatch`
  - Extend `HeterDispatchPolicy._assign` / `should_skip_group` signatures with
    optional `forward_batch` kwarg
  - Register `"dyna_thresh"` in `_POLICY_REGISTRY`
- `python/sglang/srt/layers/moe/heter_moe.py`
  - `HeterFusedMoE.forward`: pass `forward_batch` to `policy._assign` and
    `policy.should_skip_group`
- `test/test_heter_moe/unittest/test_dispatch_policy.py` — tests 1–7 above
- `test/test_heter_moe/unittest/test_correctness.py` — test 8 above
- `test/test_heter_moe/unittest/test_efficiency.py` — test 9 above
- `test/test_heter_moe/e2e/bench_serving/heter_config_dyna_thresh.json` —
  example config alongside existing policy examples
