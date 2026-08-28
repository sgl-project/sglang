"""Row layout of the two-pool fp8 unified_kv cache, shared by its writers.

The pools are separate allocations with the same row count and one row index
addresses both, so these numbers belong with the kernels that write the rows
rather than with the pool that allocates them.
"""

# The fp8 nope row is a fixed 512 B whatever the payload: 448 B latent, then
# 14 B of E8M0 tile scales (7 tiles, each written twice -- the asm reader reads
# every tile scale twice), then 50 B nobody touches. Keep in sync with aiter's
# pack_v4_nope_scale; the 512 B stride is what the reader assumes.
DSV4_FP8_NOPE_ROW_BYTES = 512
DSV4_FP8_QUANT_TILE = 64
