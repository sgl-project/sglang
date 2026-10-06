## Cache-aware ancestor fallback

With `--policy cache_aware_zmq --cache-aware-ancestor-fallback`, a matched node without owners falls back to
its deepest matched ancestor that still has owners. `--cache-threshold` is
applied to that owned prefix, not the longer structural path. Existing
worker eligibility, queue gates, and storage-tier preferences still apply.
The flag defaults to off. Omit it to retain deepest-node-only routing in the
same image; changing it requires restarting the router. Diagnostics work in
both modes.

Structural overlap/query counters keep their meaning.
`diverted_overlap_blocks` now reports owned depth rather than structural depth
when ancestor fallback is enabled; these differ for unowned suffixes. Two histograms
with the `model_id` label distinguish recorded node ownership from empty paths:

- `sgl_router_owned_overlap_blocks`: deepest owned prefix before threshold
  and load filtering.
- `sgl_router_ancestor_fallback_blocks`: structural suffix skipped when an
  ancestor owner is selected. Its `_count` counts those selections; `_sum`
  counts skipped blocks, **not** recovered engine cache hits.

Ordinary affinity selections that fall back to an ancestor use
`decision="ancestor_hit"` for the decision counter and all three
query/matched/selected block counters. Queue and admission outcomes keep their
existing decision labels; the ancestor histogram can also include those picks.
The label describes the eligible affinity captured during selection, even if
a later metric lookup in another hash mode reports a different overlap.

Update Grafana filters from `decision="cache_hit"` to
`decision=~"cache_hit|ancestor_hit"` wherever the panel should include both
ordinary affinity outcomes. For their selected/query ratio, apply the same
filter to numerator and denominator:

```promql
sum(rate(sgl_router_selected_overlap_blocks_total{decision=~"cache_hit|ancestor_hit"}[5m]))
/
sum(rate(sgl_router_cache_aware_query_blocks_total{decision=~"cache_hit|ancestor_hit"}[5m]))
```

Apply the same model/endpoint scope to both sides. For the overall predicted
hit ratio, include **all** decisions in both sides instead. This example only
measures ordinary affinity selections.

If structural overlap clears the threshold but owned overlap does not, the
decision remains `matched_node_unowned`, even when a shallower owner exists.
Use `owned_overlap_blocks` to assess whether an absolute-length threshold is
needed. Queue diversions report owned depth in `diverted_overlap_blocks`.

Node ownership still does not require continuous ownership along the path.
`sgl_router_owner_path_gap_total{model_id}` counts a sampled selection once
when any hash mode returns at least one deepest-node owner missing an ancestor
on all tiers. It includes below-threshold and unselected owners, so it is a
diagnostic of the index, not a count of misrouted requests. Worker identity
includes DP rank; mixed host/device ownership for the same worker is valid.
Checks run only with a metrics sink, once per 128 eligible lookups.
`sgl_router_owner_path_gap_samples_total` counts sampled selections; divide the
gap counter rate by this counter rate to estimate their fraction. Counts are
unweighted samples, not an exact total. Both counters count `select` calls,
like decision counters: retries may count the same request more than once.
Each sampled check walks parent links and stops at the first gap; sampled
lookups still have extra read-lock time. Unsampled lookups do not scan
ancestors. Continuous-owner filtering is deferred; a positive counter means
the reported owned/selected depth may overstate the usable prefix.

Use `sgl_router_selected_overlap_blocks_total` for the selected destination's
prefix depth. Actual cache-hit and TTFT gains require an engine replay A/B;
router ownership does not guarantee that an SWA window is still available.

Unowned interior nodes remain in the tree while they have descendants. This
fallback does not establish why their ownership disappeared or reconcile
stale owners. Event continuity, restart cleanup,
and occupancy accounting need separate verification; an ancestor owner can
also be stale. The optional
absolute-length affinity threshold is deferred pending measured owned-depth
distributions. Roll out router changes separately after engine configuration
changes stabilize so their effects can be distinguished.

## Local lookup benchmark

Release build with Rust 1.90, 2,500 blocks, 200 warmups and 2,000 timed
lookups per case. Five interleaved runs; table entries are medians of the
per-run mean latency in microseconds:

| Owners | Base d4637ea | Fallback on, no diagnostic | Fallback on, 1/128 sampled | Fallback off, no diagnostic |
|---|---:|---:|---:|---:|
| 1 | 43.6 | 43.6 | 44.1 | 43.6 |
| 3 | 43.4 | 44.6 | 44.8 | 43.9 |
| 6 | 43.8 | 44.2 | 46.0 | 43.9 |

The sampled column invokes the diagnostic every 128 lookups. It measures tree
lookup cost, excluding the policy's sampling counter, token hashing, metric
emission, concurrent KV-event writers, and engine work. It does not establish
router throughput, writer tail latency, or Step5 cache-hit gains. Sampled
lookups still perform the full continuity scan.

Reproduce the current modes with:

```sh
cargo run --release --example long_chain_match
cargo run --release --example long_chain_match -- sampled
cargo run --release --example long_chain_match -- disabled
```
