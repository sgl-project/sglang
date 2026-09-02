# DSV4 Attention Capability Matrix

This folder tracks DeepSeek-V4 attention tests. DSV4 has method-specific
sparse/indexer metadata and a packed FP8/BF16 KV cache layout, so it is not
folded into the dense, MLA, or DSA folders. The single attention backend
here is `dsv4` (which dispatches through `flash_mla`); the rows below
distinguish the **`compress_ratio` mode** that each test exercises.

## Coverage Matrix

Columns are runner modes; rows are `compress_ratio` modes of the single
`dsv4` backend. Cells use:
- **✓ \<variants\>** — exercised, with the config variants listed in the cell
- **—** — not applicable / not exercised
- **production-unreachable: \<reason\>** — production never invokes this
  combination, so the test runner asserts against it at the call site
- **blocked: \<reason\>** — would crash on a hard assertion if attempted;
  also asserted against at the call site
- **deferred: \<reason\>** — could land later, currently disabled

| `compress_ratio` | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `0` (SWA-only) | ✓ EXTEND no-prefix / prefix-within-window / nonzero `attn_sink` / above-window / seq_len==SWA_WINDOW / seq_len below-page / seq_len at-page / seq_len above-page / prefix-exact-page / total-exact-page + DECODE within-window / multi-request / above-window | ✓ DECODE within-window + multi-request | — | — | ✓ EAGLE chain (topk=1) `prefix_lens=(64,96)` | ✓ EAGLE chain CG `prefix_lens=(64,96)` | ✓ EAGLE ragged-accept | ✓ EAGLE uniform `extend_lens=(4,4)` | — | ✓ chain `prefix_lens=(32,64)`, `num_steps=3` (`DeepseekV4MultiStepBackend` capture/replay vs. per-step-init eager) | ✓ uniform `extend_lens=(4,4)`, `prefix_lens=(64,96)` (production `EAGLEDraftExtendCudaGraphRunner` through `_create_dsv4_prefill_backend`; uses loose `DSV4_GRAPH_ATOL=1e-1` and skips strict `topk_index` exact-match to absorb CG accumulation drift) | — |
| `4` (C4) | ✓ EXTEND `prefix_lens=(64,)`, `extend_lens=(16,)` + DECODE `prefix_lens=(64,)` (extra K cache written directly via `set_extra_key_buffer`; `c4_sparse_page_indices` seeded manually because indexer is bypassed) | ✓ DECODE `prefix_lens=(64,)` | — | — | ✓ EAGLE chain (topk=1) `prefix_lens=(64,96)` | ✓ EAGLE chain CG `prefix_lens=(64,96)` | production-unreachable: draft layer is SWA-only | production-unreachable: draft layer is SWA-only | — | — | — | — |
| `128` (C128) | ✓ EXTEND `prefix_lens=(128,)`, `extend_lens=(16,)` + DECODE `prefix_lens=(128,)` | ✓ DECODE `prefix_lens=(128,)` | — | — | ✓ EAGLE chain (topk=1) `prefix_lens=(128,160)` | ✓ EAGLE chain CG `prefix_lens=(128,160)` | production-unreachable: draft layer is SWA-only | production-unreachable: draft layer is SWA-only | — | — | — | — |

## Input And Config Coverage

- `num_heads=64` (matches DSV4 production; `flash_mla.sparse_decode_fwd`
  constrains `h_q` to specific values like 16/32/64/128).
- DeepSeek-V4 shape metadata: `qk_nope_head_dim=448`, `qk_rope_head_dim=64`,
  `kv_lora_rank=448`, `head_dim=512`.
- `page_size=256` (the DSV4 backend asserts this exactly —
  `deepseek_v4_backend.py:355`, `dsv4/metadata.py:134`). Per-page-boundary
  coverage uses this hardcoded page size: `seq_len=255` (one below page),
  `seq_len=256` (exactly one page), `seq_len=257` (one above page),
  `prefix_lens=256+extend_lens=4` (prefix equals one page), and
  `prefix_lens=240+extend_lens=16` (prefix + extend exactly equals one
  page). `seq_len=128` covers the SWA-window-boundary `seq_len ==
  SWA_WINDOW` case. The fixture auto-scales `max_context_len` for the
  larger sequences so `req_to_token` has room.
- Packed FP8 nope + BF16 rope SWA cache layout (584 bytes/token) comes from
  `DeepSeekV4TokenToKVPool`.
- SWA window = 128 (`SWA_WINDOW` constant in `deepseek_v4_backend.py:67`).
- Tolerance is held loose (`DSV4_ATOL = DSV4_RTOL = 5e-2`) to absorb
  `flash_mla` FP8 GEMM accumulation variance against the dequantized reference.

## Reference Implementation Notes

- The reference is a **vanilla PyTorch softmax** over the projected BF16 K
  the fixture stashes on `fixture._swa_bf16_k_per_req` (and
  `fixture._extra_bf16_k` for the C4/C128 cases). It does NOT read bytes
  back from the production cache — that would couple the test to
  `quant_to_nope_fp8_rope_bf16_pack_triton` / `set_swa_key_buffer_radix`
  and a silent pack/write bug would corrupt both paths identically. The
  vanilla BF16 K diverges from the FP8-dequantized K that `flash_mla`
  reads by the FP8 quant noise; the `DSV4_ATOL = DSV4_RTOL = 5e-2`
  tolerance absorbs that (graph-replay cases use a slightly looser
  `DSV4_GRAPH_ATOL = 1e-1` to absorb the additional accumulation drift
  introduced by `use_prefill_cuda_graph=True` padding).
- For C4/C128, the reference reads the upgraded `DSV4AttnMetadata`'s
  per-q-token `swa_page_indices` / `c4_sparse_page_indices` /
  `c128_page_indices` to learn which entries the kernel attends to. The
  reference rebuilds metadata for the current batch on every call (the
  speculative graph runner invokes `expected_output` before
  `init_forward_metadata*`) and reseeds `c4_sparse_page_indices` after
  `on_after_cuda_graph_warmup` so it observes the same indices the
  backend forward saw.
- The attention-sink correction is applied by appending a virtual key with
  per-head score `attn_sink` and value `0`. With the default
  `attn_sink_value=-1e30` this is a numerical no-op; the
  `dsv4_swa_extend_nonzero_attn_sink` case exercises the correction with
  `attn_sink_value=0.0`.

## Production-Unsupported

- **`compress_ratio in {4, 128}` + `DRAFT_EXTEND` (eager OR CUDA-graph)** —
  *production-unreachable*, not "broken". The DSV4 draft model
  (`deepseek_v4_nextn.DeepseekV4ModelNextN`) is a single decoder layer
  built with `compress_ratio_override=COMPRESS_RATIO_NEXTN_LAYER = 0`
  (`python/sglang/srt/models/deepseek_v4_nextn.py:47,105`), which flows
  through `MQALayer.__init__` at `deepseek_v4.py:232-237` and forces the
  draft layer to SWA-only regardless of `config.compress_ratios`.
  Production therefore never invokes `forward(compress_ratio=4 or 128,
  forward_mode=DRAFT_EXTEND)`; the target model uses C4/C128 only in
  DECODE / TARGET_VERIFY paths (which DO populate the C4/C128 metadata
  via `need_compress=True`). If a test were to attempt the combination,
  `init_forward_metadata_draft_extend` at `deepseek_v4_backend.py:636-663`
  hardcodes `need_compress=False`, leaving `c4_sparse_page_indices` /
  `c128_flashmla_metadata` at None and `forward(compress_ratio=4)` would
  trip `extra_indices.shape[-1]` / `forward(compress_ratio=128)` would
  trip a flash_mla `tile_scheduler_metadata` assert. The runner asserts
  `case.compress_ratio == 0` at the call site for both
  `run_dsv4_draft_extend_attention_case` and
  `run_dsv4_eagle_draft_extend_cuda_graph_case` to make this unreachable
  state loud at the test level.
- **MTP `topk > 1`** — `deepseek_v4_backend.py:369` asserts `self.topk in [0, 1]`.
  Same in the HIP radix variant (`deepseek_v4_backend_hip_radix.py:363`). DSV4
  speculative draft-extend / target-verify is *always* chain (`topk=1`);
  tree spec is structurally impossible. **DE-V2 CG, EAGLE-draft tree runner,
  EAGLE-DE tree runner, FKVMTP runner** are therefore "—" not "deferred".
- **Non-256 page size** — `deepseek_v4_backend.py:355` (and HIP radix variant
  `:349`, `dsv4/metadata.py:134`) asserts `page_size == 256`.
- **Non-512 head_dim** — `deepseek_v4_backend.py:345-347` asserts
  `head_dim == 512`. DSV4 is hard-wired to `qk_nope=448 + qk_rope=64`.
- **Unknown `compress_ratio`** — `DSV4AttnMetadata.get_flashmla_metadata`
  raises `ValueError(f"invalid {compress_ratio=}")` for anything outside
  `Literal[0, 4, 128]` (`deepseek_v4_backend.py:125-133`).
- **Forward modes outside the `_GraphBucket` set** —
  `deepseek_v4_backend.py:320-328` raises `NotImplementedError` for anything
  not in `{decode_or_idle, target_verify, draft_extend(v1 or v2)}`. Same in
  `init_forward_metadata` at `deepseek_v4_backend.py:713-714`. PCG/BCG
  split-op extend is therefore structurally unreachable.

## Compressor / C4Indexer — intentionally out of scope for this matrix

`Compressor` and `C4Indexer` are `nn.Module` instances owned by the DSV4
**model** (`models/deepseek_v4.py:296-311`), not by the attention backend.
The model's forward calls `self.indexer(...)` and
`attn_backend.forward_core_compressor(x, ..., self.compressor)` *before*
attention; their only outputs that flow into the attention backend are:

- **Compressor**: writes bytes into `extra_k_cache` at the
  `c4_out_loc` / `c128_out_loc` positions. The locations come from the
  backend's `init_compression_metadata` Triton kernel
  (`deepseek_v4_backend.py:182`), not from the Compressor.
- **C4Indexer**: writes the `c4_sparse_page_indices` field that the
  backend's `forward_extend` / `forward_decode` then read.

The attention backend's contract with both is purely: "I gave you a place
to write; you wrote something there; I'll read what you wrote." The
current fixture verifies exactly that contract by supplying known-good
synthetic bytes/indices through the **same production pack + store path**
(`quant_to_nope_fp8_rope_bf16_pack_triton` + `set_extra_key_buffer` at
`common/attention_methods/dsv4_attention.py:1193-1195`) and stashing the
unquantized BF16 K on the fixture for the reference. The
`init_compression_metadata` Triton kernel that produces page metadata IS
exercised; what's skipped is only the Compressor and C4Indexer
**`nn.Module` forward math** (`x → compressed_kv` and
`x, q_lora → page_indices`).

Compressor / C4Indexer math correctness belongs at the **component
level** — `test/srt/test_dsv4_compressor.py` and
`test/srt/test_dsv4_c4_indexer.py` are the natural homes, against
pure-PyTorch references of those modules' math. Same rationale as why
RoPE is out of scope for the attention-backend matrix (PLAN.md "RoPE
handling"): pre-processing modules whose outputs are inputs to the
attention backend.

## Next Work

- Component-level Compressor / C4Indexer correctness tests at
  `test/srt/` (separate from this matrix). Optional — the attention
  backend already verifies its end of the contract via known-good
  synthetic inputs.
