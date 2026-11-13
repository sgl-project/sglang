# Replay admission audit

The replay path borrows tree-owned Full KV. GDN checkpoint positions do not
constrain which already-cached Full KV values can be read.

Removed admission restrictions:

- An interior GDN checkpoint inside the replay chunk.
- A zero GDN prefix, a separate GDN checkpoint-grid alignment requirement, and
  the assumption that the cache must contain exactly two components.
- BF16-only, TP=1, PP=1, and extra-buffer-only gates. Projection slicing uses
  native rank-local unquantized weights; KV ownership is independent of the
  state-buffer strategy. Native no-buffer caching still requires page size 1.
- Disabling reads merely because request cache insertion is disabled, or because
  session caching is enabled for other requests.
- Disabling the whole server when LoRA is enabled but no request uses an adapter.
- Repeated model assertions already implied by model construction/admission,
  the repeated fresh-index assertion, and the redundant borrowed-slice length
  assertion after a successful Full prefix lookup.

The interior-checkpoint change also requires a lifecycle correction: inserting
GDN state at 7936 must not shrink the protection of borrowed KV through 8192.
The extra Full pin stays held until the normal checkpoint lock covers that
entire prefix. Teardown releases it after request-owned KV is freed.

Remaining conditions correspond to operations this implementation needs:

| Condition | Reason |
|---|---|
| Enabled UnifiedRadixCache with Mamba state | Supplies the prefix lookup, Full pin, and checkpoint/release lifecycle used here. |
| Replay-aware model forward and FlashInfer | These paths compute compact K/V and write only the selected fresh rows. Other model/backend implementations do not consume this metadata. |
| Unquantized weights; no active LoRA in the batch | Raw weight slicing bypasses quantized projection methods and LoRA updates. An ordinary request with LoRA in a mixed batch would also lose its updates. |
| No mixed decode/prefill, speculation, prefill graphs, CP/DCP or attention DP | These paths add, reorder, shard, or capture token rows; the fresh-row metadata has not been transformed for those layouts. Removing the TP gate must not implicitly admit DCP's different slot/index layout. |
| No HiCache/disaggregation or active session | The lookup/pin covers a local device tree, not asynchronous transfer dependencies or session-owned records. |
| No multimodal/embedding/position overrides | The rematch uses plain token IDs; these inputs require their native cache-key semantics. |
| Nonempty, page-aligned chunk; matching protected prefix; no pending borrowed pin | Preserves page ownership and the request's current borrow. In particular, borrowing a partial final page could let subsequent decode overwrite tree-owned slots. |
| Full device hit reaches the chunk end | The current per-request path borrows the entire chunk. A cached prefix plus uncached suffix within one request still needs a separate partial-reuse implementation. |
| Rollback and terminal unpin | Allocation/table-write failures must free fresh slots only and release every temporary pin. |

Focused numerical results and compatibility checks are recorded separately from
historical latency measurements. Removing these gates does not turn old
8201-token benchmark results into pure-8192 measurements.

FP16 serving must also use the native convolution-state dtype configuration,
`SGLANG_MAMBA_CONV_DTYPE=float16`. The removed gate did not fix an invalid
model/conv-state dtype combination. The FP16 functional check uses ordinary
CUDA MM; the fixed-order MM oracle is used with BF16 weights and FP32 SSM.

## Validation

See `guard_audit_results.json` for the recorded source hashes and results:
three models at concurrency 1/8/16, cold GDN state, continuation decode,
FP16, no-buffer, inactive LoRA/Session, TP=2, PP=2, and three successive
warm reads. The pure-input cases reuse the entire 4096-token KV region.
The fixed-MM FP32-state comparisons are exact; ordinary CUDA smoke checks
make no bitwise-equivalence claim. These are functional regressions, not
new latency measurements.
