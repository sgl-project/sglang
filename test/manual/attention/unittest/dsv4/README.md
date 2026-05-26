# DSV4 Attention Capability Matrix

This folder tracks DeepSeek-V4 style attention tests. DSV4 has method-specific
sparse/indexer metadata, so it should not be folded into the dense or MLA
folders.

## Current Matrix

| Backend family | Phase 2: method correctness | Phase 3: runner compatibility | Phase 4: speculative modes | Status |
|---|---|---|---|---|
| DSV4/DeepSeek-V4 attention backends | Not implemented | Not implemented | Not implemented | Needs a small DSV4-specific fixture and independent reference. |

## Current Progress

- No default-discovery DSV4 unit test is enabled yet.
- The folder exists to keep DeepSeek-V4 sparse/indexed metadata separate from
  dense, MLA, and DSA assumptions.
- The CUDA backend hard-codes `head_dim=512`, `page_size=256`, and the
  DeepSeekV4 packed KV layout (`qk_nope_head_dim=448`,
  `qk_rope_head_dim=64`, FP8 nope plus BF16 rope), so the compact dense/MLA
  unit fixtures cannot be reused safely.
- Hardware- and kernel-specific backend paths should be added with explicit skip
  gates rather than silently widening the default suite.

## Required Fixture Work

- Model the DSV4 indexer/KV-cache layout directly instead of reusing dense Q/K/V helpers.
- Add a PyTorch reference for the selected sparse/indexed attention behavior.
- Identify local hardware gates for B200/GB300-specific paths before adding default-discovery tests.
- Account for FlashMLA attention-sink semantics and packed FP8/BF16 cache
  quantization in the reference instead of comparing against an unquantized dense
  approximation.

## Next Work

- `dsv4/test_<backend_name>.py` for a locally runnable backend or a hardware-gated backend file with clear skips.
