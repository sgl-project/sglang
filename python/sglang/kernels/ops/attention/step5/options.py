# SPDX-License-Identifier: Apache-2.0
# Former SGLANG_OPT_STEP5_* / SGLANG_STEP5_* settings, frozen at their defaults.
# None selects by hardware at the read site.

ROUTER_PADDING = True
# Limits count actual GEMM token rows (including graph padding), not requests.
# Set a limit to 0 to disable that overlap path.
SHARED_OVERLAP_MAX_ROWS = 256
HEAD_GATE = True
MOE_FINALIZE_SHARED = None
SWA_FA4_GATE_EPILOGUE = True
SWA_FA4_PDL = True
# 0 = heuristic (hd192 paged decode s=5 is ~1.7 us/layer slower than s=1),
# 1 = off (default), n = force. Deterministic inference still forces 1.
SWA_FA4_NUM_SPLITS = 1
# 0 = keep FA4 default tile_n=128 (cp.async paged on page64).
# 64 = TMA paged (page_size == tile_n). Decode / verify / draft-extend only.
SWA_FA4_TILE_N = 0
# Fuse SWA QK-norm/RoPE with the paged KV-cache store.
SWA_FUSED_QKNORM_STORE = True
# Shift paged KV origin to the SWA window (cache_leftpad + reduced
# seqused_k) so tile_n=128 decode drops the unused first n-block.
SWA_FA4_WINDOW_ALIGN = True
SPARSE_PREFILL_SM100 = True
FOLD_INDEXER_SCALE = None
SCORING_CUTE = True
TOPK_DEEPSELECT = True
TOPK_DEEPSELECT_DECODE = True
SCORING_PREFILL_SM100 = True
FUSE_INDEXER_FP8_CAST = None
INDEXER_WEIGHT_FIRST = True
# Fuse norm into the scoring layout for eligible decode/verify forwards.
INDEXER_NORM_PACKED = True
# Query rows per DSA selection tile; 0 sizes the tile from the 512 MiB FP32
# score budget alone, 2048 restores the fixed tile.
SELECTION_TILE_ROWS = 0
