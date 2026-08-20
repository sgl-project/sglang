"""Packed schedule ABI: one int64 for each work tile.

The host builder writes the three fields and the device decoder reads them. The
fields use 63 bits, not 64. A torch int64 is signed, so bit 63 must stay clear.
A full entry otherwise becomes a negative number.

The earlier version packed the fields into an int32 with 10, 10 and 12 bits.
That version allowed only 8192 rows for each expert. A long chunked prefill
with uneven routing reaches that limit. The wider fields use more workspace
memory. At a chunk of 4096 tokens, each stage grows by 238 KiB with 32 experts
and hidden size 4096. Each stage grows by 510 KiB with 48 experts and hidden
size 6144.
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
