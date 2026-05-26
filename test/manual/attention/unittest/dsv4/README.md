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
- **blocked: \<reason\>** — production-unsupported, not a follow-up
- **deferred: \<reason\>** — could land later, currently disabled

| `compress_ratio` | Eager Phase 2 | CG decode | PCG extend | BCG extend | Verify eager | Verify CG | DE eager | DE CG | DE-V2 CG | EAGLE-draft runner | EAGLE-DE runner | FKVMTP runner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `0` (SWA-only) | ✓ EXTEND no-prefix / prefix-within-window / nonzero `attn_sink` / above-window EXTEND + DECODE within-window / multi-request / above-window | ✓ DECODE within-window + multi-request | — | — | ✓ EAGLE chain (topk=1) `prefix_lens=(64,96)` | — | ✓ EAGLE ragged-accept | — | — | — | — | — |
| `4` (C4) | ✓ EXTEND `prefix_lens=(64,)`, `extend_lens=(16,)` + DECODE `prefix_lens=(64,)` (extra K cache written directly via `set_extra_key_buffer`; `c4_sparse_page_indices` seeded manually because indexer is bypassed) | ✓ DECODE `prefix_lens=(64,)` | — | — | ✓ EAGLE chain (topk=1) `prefix_lens=(64,96)` | — | — | — | — | — | — | — |
| `128` (C128) | ✓ EXTEND `prefix_lens=(128,)`, `extend_lens=(16,)` + DECODE `prefix_lens=(128,)` | ✓ DECODE `prefix_lens=(128,)` | — | — | ✓ EAGLE chain (topk=1) `prefix_lens=(128,160)` | — | — | — | — | — | — | — |

## Input And Config Coverage

- `num_heads=64` (matches DSV4 production; `flash_mla.sparse_decode_fwd`
  constrains `h_q` to specific values like 16/32/64/128).
- DeepSeek-V4 shape metadata: `qk_nope_head_dim=448`, `qk_rope_head_dim=64`,
  `kv_lora_rank=448`, `head_dim=512`.
- `page_size=256` (the DSV4 backend asserts this exactly —
  `deepseek_v4_backend.py:355`, `dsv4/metadata.py:134`).
- Packed FP8 nope + BF16 rope SWA cache layout (584 bytes/token) comes from
  `DeepSeekV4TokenToKVPool`.
- SWA window = 128 (`SWA_WINDOW` constant in `deepseek_v4_backend.py:67`).
- Tolerance is held loose (`DSV4_ATOL = DSV4_RTOL = 5e-2`) to absorb
  `flash_mla` FP8 GEMM accumulation variance against the dequantized reference.

## Reference Implementation Notes

- The reference reads the same SWA cache buffer that the backend wrote and
  dequantizes it byte-for-byte (FP8 nope × UE8M0 scale + BF16 rope) — this is
  necessary because FP8 quantization is the production path and the test
  cannot independently produce the same packed bytes.
- For C4/C128, the reference reads the upgraded `DSV4AttnMetadata` to learn
  which slots the kernel will attend to, then unpacks and softmaxes those
  exact slots. This couples the reference to the production metadata builder
  (intentionally — the compressor itself is bypassed).
- The attention-sink correction is applied by appending a virtual key with
  per-head score `attn_sink` and value `0`. With the default
  `attn_sink_value=-1e30` this is a numerical no-op; the
  `dsv4_swa_extend_nonzero_attn_sink` case exercises the correction with
  `attn_sink_value=0.0`.

## Production-Unsupported

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

## Next Work

- Add Verify CUDA-graph capture/replay for SWA + C4 + C128 once the lazy
  metadata upgrade and `c4_sparse_page_indices` seeding are stable across
  capture+replay.
- Add EAGLE `DRAFT_EXTEND` CUDA-graph runner coverage (eager DE is enabled).
- Optional: model the `Compressor` and `C4Indexer` paths so the C4/C128 cases
  no longer bypass them. Today only the `extra_k_cache + extra_indices`
  integration into `flash_mla` is verified, not the compressor math itself.
