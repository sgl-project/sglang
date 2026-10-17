# MiniMax-M3 mono kernels (gfx950, TP4)

One launch for each sparse MoE layer of a decode or linear EAGLE verify step of 1..16 tokens: K1 (input add + norm,
the fused qkv / index projection, q / k norms, RoPE, the K / V / index-K cache inserts, the indexer scores) fused in
front of K4 (top-k block selection, sparse attention, o_proj and its all-reduce, router, the shared and routed MXFP4
experts and the MoE all-reduce). The host side is `sglang.srt.models.minimax_m3_mono` (the mono spec); enable with
`--layer-backend mono`.

## Origin

Vendored from [ROCm/aiter PR6047](https://github.com/ROCm/aiter/pull/6047) @ e94e7613526a29e4f72e1462a886eaafae04ab7f
(MIT, header kept in every file), with the cache addressing of aiter
[7edb5dcaf9](https://github.com/ROCm/aiter/commit/7edb5dcaf99ae02d2c9a30e76669d9ba9985cc42) (the cache page address in
the 64-bit buffer base, so accesses past 2 GiB keep a page-local lane offset; its host API changes are not taken).

## Local changes

- One build configuration: K1 fused into K4, one index head a rank, separate K and V planes (page16 SHUFFLE) with the
  layer's fixed scales, a bf16 router weight. Removed: the separate K1 / K4 launches, the vLLM K | V block layout,
  ATOM's per-token K / V scales, the fp32 router weight and the `sparse_table` / `sel` scratch outputs. Indexer
  context parallelism (`IndexHeads.count > 1`) is kept in the device code; the spec does not build it yet.
- Top-k block selection follows ROCm/ATOM's M3 select stage @ 7fc0926c6 (MIT): a floor from the lane groups' maxima,
  then only the blocks at or above it are counted against each other (`common/device/select.py`), short and long
  contexts alike. The candidates borrow K4's `p8` LDS array and a long context's entries `red`, so the LDS size is
  unchanged (159.8 of 160 KB at 16 tokens).
- Device helpers come from `../common`: the tagged `Mailbox` with the step epoch as its tag (`step_tag`, nothing
  cleared between steps), peer-buffer hand-offs as named `preg` regions (system scope), scratch ones as raw addresses
  (device scope). `device.py` holds what `common` does not: `step_tag`, `release_stores`, `ar_pack_sumsq`.
- `reduction.py` keeps the 6144-wide all-reduce norm order of the unfused path, so rstd stays bit-exact with it.
