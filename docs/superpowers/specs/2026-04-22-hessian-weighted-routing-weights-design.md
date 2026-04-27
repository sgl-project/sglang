# Hessian-weighted routing-weight dispatch policy

Augment the runtime `total_weight` dispatch policy by multiplying each
expert's summed routing weight by a static per-expert *importance* derived
offline from Hessian + first-order sensitivity. Land it as a new variant
(`hess.json`) in the existing sweep. Keep the `hot*` / `thr*` variants on
disk but commented out so the sweep focuses on the new policy.

## Motivation

`TotalWeightHeterDispatch` already ranks experts by summed routing weight
and places the top-K in BF16. This ignores the offline signal from
`hessian_scores.json`, which says *how much end-to-end loss is hurt if this
expert is forced to INT4*. Two experts with equal routing weight should
not get equal treatment if one is hessian-critical and the other is noise.

The static heter-assignment (`policy/heter_assign/assign_experts.py`)
already uses hessian scores to pick the *pool* of dual-precision experts.
Within that pool, the runtime policy still decides who gets BF16 per-batch.
This spec bridges the two so the runtime score becomes:

```
score(E) = importance(E) · total_routing_weight(E)
```

where `importance(E)` is non-negative, zero whenever hessian is at or below
the first-order noise floor, and clipped at 0 for negative (INT4-neutral)
scores. If every expert in a layer falls below the noise floor, that
layer's importance degenerates to all-ones (equivalent to pure
`total_weight`) so assignment does not collapse.

## Artifacts

### `expert_importance.json` (new, written by `gen_heter_configs.py`)

Per-layer importance arrays, same per-mc directory as `int4_only_experts.json`:

```json
{
  "0": [0.0, 1.23e-3, 0.0, 8.9e-4, ...],   // length = num_experts
  "1": [...],
  ...
}
```

- Non-negative `float32` values.
- `0.0` means hessian ≤ `|fo|_mean` (or hessian negative).
- If a whole layer would otherwise be all zeros, array is set to all `1.0`.
- Coverage: one entry per layer × expert; validated on write.

### `heter_config.json` gains one field

```json
{
  ...,
  "int4_only_experts_file": "/abs/path/mc8/int4_only_experts.json",
  "expert_importance_file": "/abs/path/mc8/expert_importance.json"
}
```

Absolute path, same convention as the existing `int4_only_experts_file`.
Only written when ranking is `hessian`. Optional: policies that do not
need importance ignore it.

### `hess.json` variant (new, written by `gen_dyna_variants.py`)

One per mc, alongside existing variants:

```json
{
  "groups": [...],
  "policy": "hessian_weighted_routing_weights",
  "int4_only_experts_file": "...",
  "expert_importance_file": "..."
}
```

Inherits base `group_size_ratios`. No `policy_params`.

## Policy implementation

### `HessianWeightedRoutingWeightsDispatch` (in `python/sglang/srt/layers/moe/heter_policy.py`)

```python
class HessianWeightedRoutingWeightsDispatch(HeterDispatchPolicy):
    """score(E) = importance(E) × sum(token_final_scales for E)."""

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        importance: torch.Tensor,  # [num_experts], non-negative float32
        fallback_seed: int = 42,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: int = 1,
    ):
        super().__init__(...)
        assert importance.shape == (num_experts,)
        self._importance = importance.to(device=self._device, dtype=torch.float32)
        self._fallback = RandomHeterDispatch(...)
        self._weight_sum_buf = torch.empty(
            num_experts, device=self._device, dtype=torch.float32)

    def _assign(self, token_selected_experts, token_final_scales):
        if token_selected_experts is None or token_final_scales is None:
            return self._fallback._assign(token_selected_experts, token_final_scales)

        buf = self._weight_sum_buf
        flat_experts = token_selected_experts.reshape(-1).long()
        flat_scales = token_final_scales.reshape(-1)
        buf.zero_()
        buf.scatter_add_(0, flat_experts, flat_scales)
        buf.mul_(self._importance)

        return _assign_by_score_gpu(
            buf, self._num_experts, self._group_size_ratios,
            self._expert_to_group_buf, self._group_labels)
```

Register as `"hessian_weighted_routing_weights"` in `_POLICY_REGISTRY`.

### Threading importance through `HeterFusedMoE`

- `HeterFusedMoE.__init__` gains `layer_id: Optional[int] = None`.
- When `heter_config` contains `"expert_importance_file"`, `__init__`:
  - Requires `layer_id is not None` (raises `ValueError` otherwise).
  - Opens the JSON, extracts the `str(layer_id)` entry, validates length
    == `num_experts`.
  - Builds a `torch.tensor([num_experts], dtype=float32, device=self.device)`,
    stores as `self.expert_importance` (VRAM-resident buffer).
  - Passes `importance=self.expert_importance` into `create_policy` via
    `policy_kwargs["importance"]`.
- `from_fused_moe` gains matching `layer_id` kwarg, passes through.
- `apply_heter_precision_to_model` passes the existing loop `layer_id`.

## Offline importance construction (in `gen_heter_configs.py`)

When `args.ranking == "hessian"`, after `_load_hessian_scores` and
`fo_mean = sum(|fo|) / n`, produce importance dict:

```python
importance_by_layer: Dict[str, List[float]] = {}
for L in range(num_layers):
    row: List[float] = []
    for E in range(num_experts):
        h = scores[(L, E)]
        row.append(h if h > fo_mean else 0.0)
    if all(v == 0.0 for v in row):
        row = [1.0] * num_experts
    importance_by_layer[str(L)] = row
```

Write to `mc{mc}/expert_importance.json`. The file path goes into the
`heter_config.json` emitted by `_write_outputs` as
`expert_importance_file` (absolute path).

The per-mc scope is redundant (importance is mc-invariant) but keeps the
artifact next to its sibling config files and avoids a cross-mc reference.

## Variant generation (in `gen_dyna_variants.py`)

- Add the `hess.json` emission inside the existing per-mc loop.
- Wrap the two existing `for hot_pct in HOT_PCTS:` and
  `for thr in THRESHOLDS:` loops in `if False:` blocks (or comment them
  out as whole blocks) with an explanatory comment so they can be
  re-enabled by a single edit.
- Copy `expert_importance_file` and `int4_only_experts_file` from base
  into the variant.

## Dry-run flag

Both `gen_heter_configs.py` and `gen_dyna_variants.py` gain `--dry_run`:

- Iterates per-mc as usual.
- Prints each output path and policy that *would* be written.
- Does not open any file for writing.
- Logs `|fo|_mean`, per-mc `K`, layer-level importance summaries (e.g.
  "# zero-importance experts per layer: min/median/max/...") so the user
  can sanity-check before committing to a run.

## Edge cases

- **`expert_importance_file` set but `layer_id` not plumbed** → raise
  `ValueError` at `HeterFusedMoE.__init__`.
- **Importance JSON missing a layer entry** → `KeyError` with the
  (layer_id, num_layers) context.
- **Importance row length ≠ num_experts** → `ValueError`.
- **All-zero row** → already normalized to all-ones offline; runtime
  never sees all-zero.
- **INT4-only / BF16-only mask interaction** → unchanged; masks still
  override the scored assignment after `_assign_by_score_gpu` returns
  (existing path in `_dispatch_from_expert_to_group`).
- **`token_selected_experts is None`** → delegate to `RandomHeterDispatch`
  fallback, same pattern as `TotalWeightHeterDispatch`.

## Testing

Unit tests in `test/test_heter_moe/unittest/test_dispatch_policy.py`:

1. **Identity importance (= 1) matches `TotalWeightHeterDispatch`** on
   the same routing batch.
2. **Zero-importance experts never land in BF16 group** even when they
   dominate routing weight.
3. **Mixed importance ranks by product** — two experts with equal
   routing weight rank by importance; two with equal importance rank by
   routing weight.
4. **Fallback without signals** — `_assign(None, None)` returns a valid
   shape (delegates to `RandomHeterDispatch`).
5. **Shape assertion** — passing wrong-length importance raises.

Dry-run sanity:

- `python gen_heter_configs.py --dry_run` exits 0, no files written,
  output lists every per-mc artifact path.
- `python gen_dyna_variants.py --dry_run` exits 0, no files written,
  output lists every per-mc `hess.json` path.

## Non-goals

- Changing the offline heter-pool selection (`assign_experts.py`'s
  top-K-by-hessian logic is untouched).
- Tuning the exact normalization of `importance` (raw signed hessian
  above noise floor is fine for ranking; no z-score, no log).
- Re-enabling the `hot*` / `thr*` variants — kept as commented-out code
  for quick reinstatement, not removed.
