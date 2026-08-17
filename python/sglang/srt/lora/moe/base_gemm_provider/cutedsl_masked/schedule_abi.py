"""Packed direct-schedule ABI shared by the host builder and device decoder.

One int64 per work tile: which expert, which token cluster within it, which
output cluster. The GEMM reads one word per tile instead of re-deriving the
mapping in each of its specialized warps.

The fields sum to 63 bits, not 64: torch's int64 is SIGNED, so bit 63 must stay
clear or a saturated entry packs as a negative number.

The widths put every ceiling out of reach. A tile addresses
``token_width * MAX_TOKEN_CLUSTERS`` rows per expert, so the narrowest compiled
tile (8 rows) reaches 33.5M and the widest (128) reaches 537M. The predecessor
was an int32 split 10/10/12, where those were 8192 and 131,072 -- the first of
which a long chunked prefill with lopsided routing can reach, which is why
``_token_width_for`` escalates through compiled widths.

Cost of the extra word: schedule buffers live on the LoRA backend's shared
workspace, so it is a whole-server cost. Measured at a 4096-token chunk, both
stages grow by 238 KiB on a 32-expert/4096-hidden slice and 510 KiB on a
48-expert/6144-hidden one.
"""

EXPERT_SHIFT = 0
EXPERT_BITS = 20
EXPERT_MASK = (1 << EXPERT_BITS) - 1

TOKEN_CLUSTER_SHIFT = EXPERT_SHIFT + EXPERT_BITS
TOKEN_CLUSTER_BITS = 22
TOKEN_CLUSTER_MASK = (1 << TOKEN_CLUSTER_BITS) - 1

OUTPUT_CLUSTER_SHIFT = TOKEN_CLUSTER_SHIFT + TOKEN_CLUSTER_BITS
OUTPUT_CLUSTER_BITS = 21
OUTPUT_CLUSTER_MASK = (1 << OUTPUT_CLUSTER_BITS) - 1

PACKED_BITS = OUTPUT_CLUSTER_SHIFT + OUTPUT_CLUSTER_BITS
assert PACKED_BITS == 63, PACKED_BITS

MAX_EXPERTS = 1 << EXPERT_BITS
MAX_TOKEN_CLUSTERS = 1 << TOKEN_CLUSTER_BITS
MAX_OUTPUT_CLUSTERS = 1 << OUTPUT_CLUSTER_BITS
