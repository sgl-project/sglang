# Dense Attention Capability Matrix

This folder covers standard dense MHA/GQA/MQA attention through `RadixAttention`.
Expected outputs come from independent HF-style PyTorch reference modules with
copied random projection weights, not from another SGLang attention backend.

## Coverage Matrix

Columns are runner modes; rows are attention backends. Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable (no production path for this combination)
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| Backend | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `torch_native` | ✓ full MHA/GQA/MQA input sweep + decode/extend runner-eager cases | — (no `init_cuda_graph_state` / capture / replay hooks) | — (no CG path) | — (no CG path) | deferred: extend-metadata mismatch in `TARGET_VERIFY` reference | — | — | — | — | — | — | — |
| `triton` | ✓ MHA/GQA/MQA + 10 input layouts (page 1/16/32, prefix/decode edges) | ✓ MHA/GQA/MQA decode page-boundary | ✓ MHA ragged, GQA cross-page | ✓ MHA ragged, GQA cross-page | ✓ EAGLE chain+tree, Frozen-KV-MTP chain, DFlash chain, NGRAM chain | ✓ EAGLE tree, DFlash chain, NGRAM chain | deferred: Triton `DRAFT_EXTEND` HF-ref mismatch on narrow accept layouts | — (V1 not enabled; Triton uses V2) | ✓ fixed-tokens-per-req | ✓ chain (topk=1) + tree (topk=2) | ✓ via `DRAFT_EXTEND_V2` graph runner | — (production dispatcher only wires Frozen-KV-MTP through FlashInfer-style draft backends) |
| `flashinfer` | ✓ MHA/GQA/MQA + 10 input layouts (`head_dim=64` for SM90 prefill constraints) | ✓ MHA/GQA/MQA decode page-boundary | ✓ MHA ragged, GQA cross-page | ✓ MHA ragged, GQA cross-page | ✓ EAGLE chain+tree, Frozen-KV-MTP chain, DFlash chain, NGRAM chain | ✓ EAGLE tree, Frozen-KV-MTP chain, DFlash chain | ✓ EAGLE ragged-accept, Frozen-KV-MTP ragged-accept | ✓ EAGLE ragged-accept, Frozen-KV-MTP ragged-accept | blocked: `is_draft_extend()` default `include_v2=False` → `raise ValueError` (`flashinfer_backend.py:651,748`) | ✓ chain (topk=1) + tree (topk=2) | ✓ EAGLE ragged-accept (V1) | ✓ chain (topk=1) |
| `fa3` | ✓ MHA/GQA/MQA input sweep (FA-friendly `head_dim=64`) | deferred: graph replay mismatches HF-ref | ✓ MHA ragged, GQA cross-page | ✓ MHA ragged, GQA cross-page | deferred: not enabled (EAGLE `TARGET_VERIFY` mismatches ~0.115) | deferred: same EAGLE/DFlash mismatch on replay | — | — | deferred: `DRAFT_EXTEND_V2` graph replay mismatches HF-ref (~0.618) | — | — | — |
| `fa4` | ✓ MHA/GQA/MQA input sweep (FA-friendly `head_dim=64`) | deferred: same FA graph-replay mismatch as fa3 | ✓ MHA ragged, GQA cross-page | ✓ MHA ragged, GQA cross-page | deferred: same FA mismatch | deferred: same FA mismatch | — | — | deferred: same FA mismatch | — | — | — |
| `flex_attention` | ✓ MHA/GQA/MQA input sweep | blocked: no `init_cuda_graph_state` / capture / replay hooks (`torch_flex_backend.py`) | ✓ MHA ragged, GQA cross-page | ✓ MHA ragged, GQA cross-page | blocked: no CG capture/replay path | blocked: no CG capture/replay path | — | blocked: no CG capture/replay path | blocked: no CG capture/replay path | blocked: no CG capture/replay path | blocked: no CG capture/replay path | blocked: no CG capture/replay path |
| `trtllm_mha` | ✓ decode-only MHA/GQA/MQA + page-32 boundary (prefill blocked by `Unsupported architecture`) | deferred: replay mismatches HF-ref on SM90 | — (no extend backend) | — (no extend backend) | blocked: `topk=1` only (`server_args.py:2391-2392`, `trtllm_mha_backend.py:459,492`) | blocked: same `topk=1` constraint | — | — | — | deferred: requires chain-only graph capture wiring | — | — |

### Wrapper backends (smoke tests only)

| Wrapper | Coverage |
|---|---|
| `hybrid_attn` (`prefill=triton`, `decode=flashinfer`) | ✓ EXTEND dispatches to prefill backend; ✓ DECODE dispatches to decode backend. No CG / spec coverage — the wrapper just forwards to the chosen child. |
| `tbo` (children=`[triton, triton]`) | ✓ EXTEND with no `tbo_children` set: delegates to `primary`. Sub-batched orchestration through TBO children needs scheduler-level batch splitting and is deferred. |

## Input And Config Coverage

- Page size 1, page size 16, and representative page size 32.
- Zero-prefix exact page, prefix exact page, total exact page, and page-boundary crossing.
- Ragged batches with lengths below/equal/above a page.
- Decode page-boundary batches and batch-size-1 decode.
- Attention config coverage for MHA, GQA, and MQA is separate from input-layout coverage.

## Notes on the "—" cells

- **`torch_native` graph rows** — `TorchNativeAttnBackend` does not override
  `init_cuda_graph_state` / `init_forward_metadata_capture_cuda_graph` /
  `init_forward_metadata_replay_cuda_graph`; the base class raises
  `NotImplementedError` (`base_attn_backend.py:24-55`).
- **`flex_attention` graph rows** — `TorchFlexAttnBackend` also has no CG hooks.
  It additionally rejects non-causal (`torch_flex_backend.py:151`) and cross /
  encoder-only attention (`torch_flex_backend.py:267-270`).
- **`trtllm_mha` extend rows** — backend exposes decode only; prefill currently
  reports `Unsupported architecture` and page sizes are restricted to
  `{16, 32, 64}` (`server_args.py:2849-2853`).
- **`triton` FKVMTP runner** — `FrozenKVMTPMultiStepDraftBackend` dispatch wires
  Triton through the FlashInfer-style draft path; the dedicated runner case is
  only enabled where production routes that draft worker.

## Next Work

- Debug torch-native target-verify extend metadata.
- Debug Triton `DRAFT_EXTEND` metadata/reference mismatch.
- Debug FA3/FA4 CUDA graph replay and speculative graph mismatches.
- Add backend-specific graph coverage for `trtllm_mha` once local hardware and metadata behavior allow it.
