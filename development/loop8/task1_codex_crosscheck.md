CONCLUSION 1: AGREE, narrowly.

I don’t see evidence in A-D that a `head_dim=192` specialization is needed. The write path slices `[..., :nope_dim]`, gather uses the mask’s `label_dim`, and the scoring kernel pads `label_dim` with `_next_pow2` and masks on `d_offs < label_dim`. GLM `label_dim=24` or `32` should be fine.

Caveat: this only holds if the projected layout is still `[K_noPE, V]` per head and the query tensor passed to `project_query_onto_channels` is noPE-compatible with the same prefix ordering. That is a layout assumption, not a kernel-width specialization.

CONCLUSION 2: REFUTE as stated, but agree with the core bug.

The V3.2-mask-on-GLM silent failure is real: `mask.head_dim=128` with indices `0..127` remains in range for GLM `qk_nope=192`, so exact `mask.head_dim == runtime qk_nope_head_dim` is required. `channel_selection.max() < qk_nope` is useful but insufficient because it would not catch the 128-on-192 case.

I refute the “ONE real gap” wording. Bind-time validation should also hard-check:

- `mask.label_dim == runtime/server_label_dim == channel_selection.shape[-1]`
- `channel_selection.shape[0] == runtime num_heads_local`
- `mask.num_layers` / channel-selection layer count matches runtime layer count or the DS-enabled layer map
- `self._ds_qk_nope_head_dim == self.qk_nope_head_dim`, specifically to catch the default-128 trap
- `kv_b_proj.out_features % H_local == 0`
- `kv_b_proj.out_features // H_local >= self.qk_nope_head_dim`, preferably `== qk_nope_head_dim + v_head_dim` if `v_head_dim` is authoritative there
- `channel_weights.shape == channel_selection.shape`
- keep/re-run existing `page_size`, `kv_cache_dtype`, and `index_topk` checks at bind if those runtime values are only fully authoritative there

MISSED ASSUMPTIONS:

- Excerpt B does not validate head count; if the mask has fewer heads than `H_local`, `torch.gather` can produce a smaller head dimension instead of failing immediately.
- Calibration must use `suffix_dim=v_head_dim` for projected KV extraction, not `qk_rope_head_dim`; GLM makes that distinction large: `256` vs `64`.
- The KV write hook assumes `k` is exactly the latent KV rank accepted by `kv_b_proj`, not latent+RoPE. If GLM’s hook passes `512+64`, this fails.
- Layer IDs must index the same global/local layer numbering used by the calibrated mask. GLM’s `78` layers make a DeepSeek mask likely to fail late unless checked early.
