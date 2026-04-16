# Batch-Size-Gated Heter Dispatch Policy

**Date:** 2026-04-15
**Status:** Proposed
**Owner:** huanchen

## Summary

Add a new `HeterDispatchPolicy` named `batch_size` that selects the per-forward
precision based purely on the current batch size. Below the configured token
threshold (default **128**), every expert is treated as cold (INT4); at or
above the threshold, every expert is treated as hot (BF16). The hot/cold split
is binary and global — there is no per-expert scoring.

This is a counterpart to `expert_load`: `expert_load` decides the precision
*split across experts*; `batch_size` decides the precision *for the whole
batch*.

## Motivation

Existing policies (`expert_load`, `total_weight`, `confidence`, `random`) all
require a fixed `group_size_ratios` and produce a per-expert assignment. For
workloads where small batches are latency-bound (and INT4 is competitive) and
large batches are throughput-bound (and BF16 wins), neither a fixed ratio nor a
per-expert score is the right knob — the right knob is *batch size*.

## Design

### Policy class

New class `ExpertBatchGatedHeterDispatch` in
`python/sglang/srt/layers/moe/heter_policy.py`, registered in
`_POLICY_REGISTRY` under the key `"expert_batch"`.

```python
class ExpertBatchGatedHeterDispatch(HeterDispatchPolicy):
    """Gate hot/cold by current batch token count.

    n >= threshold  -> all experts assigned to BF16 group
    n <  threshold  -> all experts assigned to INT4 group

    Per-expert scoring is not used. INT4-only experts are still
    forced to the INT4 group by _dispatch_from_expert_to_group.
    """
    def __init__(self, num_experts, group_size_ratios,
                 threshold=128, device=None,
                 int4_only_mask=None, int4_group_idx=0,
                 bf16_group_idx=None):
        super().__init__(...)
        self._threshold = threshold
        self._int4_group_idx = int4_group_idx
        # Default: BF16 group is the "other" one (binary policy)
        self._bf16_group_idx = (
            bf16_group_idx
            if bf16_group_idx is not None
            else (1 - int4_group_idx)
        )

    def _assign(self, token_selected_experts, token_final_scales):
        if token_selected_experts is None:
            # No routing info yet — default to cold
            self._expert_to_group_buf.fill_(self._int4_group_idx)
            return self._expert_to_group_buf

        n = token_selected_experts.shape[0]  # number of tokens in batch
        target = (
            self._bf16_group_idx if n >= self._threshold
            else self._int4_group_idx
        )
        self._expert_to_group_buf.fill_(target)
        return self._expert_to_group_buf
```

### Why the gate is safe under CUDA graph / torch.compile

The branch on `n` is a Python-side decision based on a tensor *shape*, not a
tensor *value*. CUDA graphs are captured per static batch size in SGLang, so
each graph capture observes a single value of `n` and bakes in one branch.
Eager mode and dynamic shapes also work because no GPU→CPU sync of tensor
contents is needed. The output `expert_to_group_buf` is filled with a constant
in-place — same tensor, fixed shape.

### Short-circuit in the layer (post-mask aware)

Update `HeterFusedMoE.forward` in `python/sglang/srt/layers/moe/heter_moe.py`
so that a group with *no live slots* is skipped entirely (no kernel call).
"Live" must be evaluated **after** `int4_only_mask` is applied, because that
mask can force experts back into the INT4 group even when the gate sent them
to BF16. The check stays host-side (no GPU→CPU sync) because both inputs to
the decision are static per-layer (mask) or shape-based (batch size).

Concretely, expose an optional
`should_skip_group(group_idx, num_tokens)` hook on `HeterDispatchPolicy`,
returning `False` by default. The hook's contract: "given this batch size,
will the *post-mask* dispatch for this group be empty?".

`ExpertBatchGatedHeterDispatch` knows enough at init time to answer this. The
base class already passes `int4_only_mask` into every policy; the new policy
caches one Python bool from it:

```python
class ExpertBatchGatedHeterDispatch(HeterDispatchPolicy):
    def __init__(self, ..., threshold=128, ...):
        super().__init__(...)
        self._threshold = threshold
        self._bf16_group_idx = (1 - self._int4_group_idx)  # binary policy
        # Static per-layer fact: do any experts live in the INT4-only mask?
        self._has_int4_only = (
            int4_only_mask is not None and bool(int4_only_mask.any().item())
        )

    def should_skip_group(self, group_idx, num_tokens):
        if num_tokens >= self._threshold:
            # Gate puts everyone in BF16. Only INT4 experts left
            # are those forced by int4_only_mask.
            if group_idx == self._int4_group_idx:
                return not self._has_int4_only
            return False
        else:
            # Gate puts everyone in INT4. BF16 group is empty,
            # int4_only_mask is a no-op (already INT4).
            return group_idx == self._bf16_group_idx
```

In `forward`:

```python
n = topk_ids.shape[0]
for group_idx, gcfg in enumerate(self.group_cfgs):
    if self.policy.should_skip_group(group_idx, n):
        continue
    # ... existing per-group dispatch
```

The existing `if self._num_bf16_experts == 0: continue` guard inside the BF16
branch becomes redundant when this hook is in place (the policy can encode it
the same way), but we will keep it as defense-in-depth for non-`batch_size`
policies and remove it only after audit.

The `int4_only_mask.any().item()` host sync runs **once at policy init**, not
per-forward. There is no runtime GPU→CPU sync.

CUDA-graph note: the skip decision is shape-based + static-bool; each captured
graph still has a fixed call structure.

### Truth table (with partial_bf16)

| `n >= threshold` | `_has_int4_only` | BF16 kernel | INT4 kernel |
|---|---|---|---|
| no  | no  | skip | run (all experts) |
| no  | yes | skip | run (all experts) |
| yes | no  | run (all experts) | **skip** ← new short-circuit |
| yes | yes | run (non-mask experts) | run (mask experts) |

### Config schema

```json
{
  "policy": "expert_batch",
  "policy_params": { "threshold": 128 },
  "groups": [
    { "name": "int4", "size_ratio": 0.0, "num_bits": 4,
      "checkpoint": "/path/to/gptq" },
    { "name": "bf16", "size_ratio": 1.0, "num_bits": 16 }
  ]
}
```

Notes:
- `group_size_ratios` is required for plumbing but unused by the gate. The
  spec keeps validation (`sum == 1.0`) untouched.
- `int4_only_experts_file` continues to work — those experts are forced to the
  INT4 group inside `_dispatch_from_expert_to_group`, regardless of the gate.
- The policy assumes exactly two groups (one INT4, one BF16). For more groups
  an explicit `bf16_group_idx` policy_param can be supplied; with N>2 groups
  the gate flips between two named indices and the others remain unused.

### Interaction with existing components

| Component | Behavior |
|---|---|
| `_int4_only_mask` | Still forces int4-only experts into `_int4_group_idx`. |
| `_bf16_id_remap` (compact BF16 tensor) | Unchanged — only consumed when the BF16 kernel runs. |
| EP / TP all-reduce | Unchanged. |
| Sentinel handling (`-1`) | Not exercised in this policy (every slot is in the active group), but still used by INT4-only remasking and by other policies. |

## Testing

New unit tests in `test/test_heter_moe/unittest/test_dispatch_policy.py`:

1. `test_batch_size_gated_below_threshold` — n=64, threshold=128 → all slots
   go to INT4 group; BF16 dispatch contains only sentinels.
2. `test_batch_size_gated_at_and_above_threshold` — n=128 and n=256 → all
   slots go to BF16 group; INT4 dispatch contains only sentinels.
3. `test_batch_size_gated_int4_only_mask_respected` — even when n=256, experts
   listed in `int4_only_mask` end up in INT4.
4. `test_batch_size_gated_custom_threshold` — threshold=64 changes the gate
   point.
5. `test_batch_size_gated_should_skip_group_no_int4_only` — without
   `int4_only_mask`: BF16 skipped below threshold, INT4 skipped at/above.
6. `test_batch_size_gated_should_skip_group_with_int4_only` — with non-empty
   `int4_only_mask`: BF16 skipped below threshold; **neither** group skipped
   at/above (INT4-only experts keep INT4 kernel live).

New end-to-end correctness check in
`test/test_heter_moe/unittest/test_correctness.py` (mirroring the existing
`expert_load` test): run a forward at n=64 and n=256 and verify outputs match
the pure-INT4 and pure-BF16 reference paths respectively.

CUDA-graph regression: extend `test_efficiency.py` to capture a graph at a
size below threshold and a separate graph at a size above threshold, and
verify both replay successfully.

## Out of scope

- Hysteresis / per-batch warmup: the gate is stateless and instantaneous.
- Dynamic threshold tuning: threshold is a config constant; future work could
  derive it from kernel benchmarks but is not part of this spec.
- More than two precision groups: not used in this policy. If needed, callers
  can set `bf16_group_idx` explicitly and accept that other groups will sit
  idle.

## Files touched

- `python/sglang/srt/layers/moe/heter_policy.py`
  - Add `ExpertBatchGatedHeterDispatch`
  - Add `should_skip_group` base method (returns False)
  - Register `"expert_batch"` in `_POLICY_REGISTRY`
- `python/sglang/srt/layers/moe/heter_moe.py`
  - Forward loop: consult `self.policy.should_skip_group(group_idx, n)` and
    `continue` when True
- `test/test_heter_moe/unittest/test_dispatch_policy.py` — new tests above
- `test/test_heter_moe/unittest/test_correctness.py` — gated end-to-end test
- `test/test_heter_moe/unittest/test_efficiency.py` — CUDA-graph capture at
  both sides of the threshold
- `test/test_heter_moe/e2e/bench_serving/heter_config_*.json` — add an example
  `batch_size` config alongside the existing `random`/`expert_load` ones
