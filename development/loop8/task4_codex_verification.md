VERDICT: **GLM-SAFE**, assuming GLM-5.1’s `kv_b_proj` output is truly packed per head as `[K_nope | V]`.

For the given GLM-5.1 MLA dims, the hooks are shape-correct: `importance` is `[num_layers, num_heads, 192]`; `q_b_proj` expects per-head width `192 + 64 = 256` and extracts the first 192 no-PE channels; `kv_b_proj` expects per-head width `192 + 256 = 448` and extracts the first 192 no-PE K channels. The K-side suffix is correctly `v_head_dim=256`, not `qk_rope_head_dim=64`, so the collected channel mask should be non-empty and width-192 for both Q and K no-PE paths.

Remaining assumption: the code still assumes DeepSeek-style MLA packing layout, specifically `kv_b_proj -> [K_nope_h0 | V_h0 | K_nope_h1 | V_h1 | ...]` and `q_b_proj -> [Q_nope_h0 | Q_rope_h0 | ...]`. If GLM packs all K heads then all V heads, or otherwise changes order, this extractor would be wrong despite matching widths. No 128-wide assumption is visible in the inlined code.
