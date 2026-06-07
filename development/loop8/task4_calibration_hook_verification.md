# Loop 8 / task4 — Calibration-hook verification for GLM-5.1 MLA shapes (AC-3, CPU portion)

**Routing:** `analyze`. Direct source verification + independent Codex confirmation
(`task4_codex_verification.md`; Codex's `exec` sandbox can't read the checkout, so the relevant
`calibrate.py` excerpts were inlined). **The mask artifact generation itself stays hardware-gated
(task5)** — this task verifies the hook *names*, *slice offsets*, and *per-head widths* are correct for
GLM's `qk_nope=192` / `v_head=256`, which is what gen-plan put in scope for the CPU side.

## Verdict: GLM-SAFE (config-driven; correct K-side suffix; covered by inheritance)

### Hook names + selection (`calibrate.py:543–649`)
- K side resolves `attn.kv_b_proj` (`k_source == "kv_b_proj"` → MLA path); Q side resolves
  `attn.q_b_proj`. Both names match the inherited `DeepseekV2AttentionMLA` projections that
  `GlmMoeDsaForCausalLM` constructs (it subclasses `DeepseekV2ForCausalLM` and overrides only
  `determine_num_fused_shared_experts`), so the hooks attach on GLM unchanged.

### No-PE slice offsets + widths (config-driven, correct for GLM)
- Dims read from config: `qk_nope_head_dim=192`, `v_head_dim=256`, `qk_rope_head_dim=64` (read
  **directly**, with the BL-20260527-mla-config-rope-dim-derivation guard — not derived from
  `hidden_size//num_heads`). `k_head_dim = qk_nope_head_dim = 192`; `importance` is `[78, H, 192]`.
- Expected packed widths gate the hooks: `full_mla_k_width = num_heads*(192+256)` (K side),
  `full_mla_q_width = num_heads*(192+64)` (Q side). The K hook raises if `kv_b_proj`'s output last-dim
  doesn't match (`"kv_b_proj output last-dim=%d does not match expected K|V"`), so a wrong width fails
  loudly rather than collecting empty/short channels (AC-3 negative test).
- **K-side suffix is `v_head_dim` (256), NOT `qk_rope` (64)** — `calibrate.py:597`
  `_extract_mla_nope_prefix(t, num_heads, k_head_dim, v_head_dim)`. Q-side suffix is `qk_rope_head_dim`
  (`:631`). This is exactly the distinction the task1 Codex cross-check flagged; it is correct here.
- `_extract_mla_nope_prefix` (`:115–134`) reshapes to `[-1, H, nope+suffix]` **before** slicing
  `[..., :nope]` (BL-20260527-reshape-before-slice-mla), so for GLM it picks the 192 K-noPE columns of
  each per-head `[K_nope(192)|V(256)]` block — never V columns of an earlier head.

### Residual assumption (covered)
Codex's only caveat: the extractor assumes DeepSeek-style **per-head interleaved** packing
(`[K_nope_h0|V_h0|K_nope_h1|V_h1|…]`). GLM inherits this exact `kv_b_proj` construction
(`deepseek_v2.py:1555` `num_heads*(qk_nope_head_dim + v_head_dim)`), so the layout holds. A
hypothetical future GLM variant that re-packed (all-K-then-all-V) would be caught at runtime by the
hook's width assertion + the new bind-time `verify_bind_shapes` `head_dim == qk_nope` check (task2).

## CPU test added
`TestMlaNopeExtractionDualShape` (`test_double_sparsity_unit.py`): sentinel-poison tests exercising
`_extract_mla_nope_prefix` directly at **both** `192/256` (GLM) and `128/128` (V3.2) for the K side
(suffix = `v_head_dim`) and `192/64` & `128/64` for the Q side (suffix = `qk_rope`), asserting the
extracted prefix is the no-PE channels (`1.0`) and never the poisoned V/RoPE columns (`100.0`).

## Still hardware-gated (task5)
The actual calibration run (offline FP8 forward of GLM-5.1 on 8×H200 to collect non-empty activations,
write the GLM-native-`label_dim` safetensors mask + content_sha256 + contract) requires the GPUs and
weights — deferred to a hardware round. This task certifies the hooks are GLM-shape-correct so that run
will collect the right channels.
