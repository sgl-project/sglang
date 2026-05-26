# DSV4 Attention Capability Matrix

This folder tracks DeepSeek-V4 style attention tests. DSV4 has method-specific
sparse/indexer metadata and packed FP8/BF16 KV cache layout, so it is not
folded into the dense, MLA, or DSA folders.

## Current Matrix

| Backend | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| `dsv4` (`compress_ratio=0`, SWA-only) | EXTEND with SWA window via the production packed FP8-nope/BF16-rope SWA cache | Not implemented | Not implemented | Drives the real `DeepseekV4AttnBackend` through `flash_mla` with `attn_sink=-1e30` and an independent PyTorch reference that dequantizes the same SWA cache. |

## Input And Config Coverage

- `num_heads=64` (flash_mla `sparse_decode_fwd` head count) with DeepSeek-V4
  shape metadata: `qk_nope_head_dim=448`, `qk_rope_head_dim=64`,
  `kv_lora_rank=448`.
- Page size and packed FP8/BF16 SWA cache layout (584 bytes/token) come from
  `DeepSeekV4TokenToKVPool`.
- `max_seq_len <= SWA_WINDOW = 128` for the SWA-only slice.
- Tolerance is held loose (`DSV4_ATOL = DSV4_RTOL = 5e-2`) to absorb flash_mla
  FP8 GEMM accumulation variance against the dequantized reference.

## Current Progress

- Phase 2 EXTEND coverage with the SWA-only path is enabled and verified
  against an independent PyTorch reference that unpacks bytes from the same
  SWA cache buffer, dequantizes with the stored UE8M0 FP8 scales, and applies
  causal + sliding-window MLA math with a virtual-key attention-sink
  correction.
- The SWA subset of `DSV4AttnMetadata` is populated manually
  (`init_compression_metadata` is skipped since C4/C128 layers are empty for
  `compress_ratios=[0]`).
- Hardware-gated / kernel-specific backend paths beyond flash_mla SWA are not
  enabled yet.

## Required Fixture Work

- Model the C4 (4x) and C128 (128x) compressed-attention layers and their
  `Compressor` / `C4Indexer` metadata. Today only `compress_ratio=0` (SWA) is
  exercised, so the compressed pipeline is not in the matrix.
- Add a decode-mode fixture that uses `DSV4RawDecodeMetadata`. EXTEND is
  covered but DECODE shares enough metadata setup to warrant a separate
  fixture.
- Add a speculative target-verify and draft-extend fixture using
  `DSV4RawVerifyMetadata`.
- Account for FlashMLA attention-sink semantics with non-zero `attn_sink`
  values (today the reference applies `-1e30`).
- Extend the reference to `seq_len > SWA_WINDOW=128` once the corresponding
  backend semantics are scoped.

## Next Work

- C4 (4x) and C128 (128x) compressed-attention coverage.
- Decode-mode SWA coverage.
- Speculative target-verify / draft-extend coverage.
- Non-zero attention-sink coverage.
- CUDA graph capture/replay for the DSV4 SWA path.
