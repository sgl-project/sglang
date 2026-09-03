# VDN-H3 block-level parity harness

Compares one MiniMax-H3 DiT block's attention between SGLang's port
(`hybrid_window_attn_h3` + `MiniMaxH3VDNLinearBranch`) and OpenVDN's reference
`HybridAttention`, on identical synthetic inputs and the real prefused weights.

1. `python sgl_block_attn.py --block 0 --shape small --out blk0.pt`
   (SGLang env; also dumps the dense base+LoRA output for the M1 smoke;
   `--shape paper` for the 345-frame workload, `--profile`, `--fp8`, `--kernel tiles`)
2. `python vdn_block_attn.py --inp blk0.pt --out blk0_vdn.pt`
   (OpenVDN venv: torch 2.13, patched diffusers, `src/` on sys.path; edit `sys.path`)
3. `python compare.py blk0.pt blk0_vdn.pt`

`VDN_H3_MATERIALIZED_DIR` points at the materialized overlay directory
(`SGLANG_DIFFUSION_CACHE_ROOT/materialized_models/OpenVDN__vdn-minimax-h3-<key>`).
Expected: relative L2 about 5e-3 to 7e-3 (bf16 rounding) for both dense and hybrid.
