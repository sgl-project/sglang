"""Packed direct-schedule ABI: one int64 per work tile (expert, token cluster,
output cluster), shared by the host builder and the device decoder.

The fields sum to 63 bits, not 64: torch's int64 is SIGNED, so bit 63 must stay
clear or a saturated entry packs as a negative number.

The predecessor was an int32 split 10/10/12, whose 8192-row-per-expert ceiling
a long chunked prefill with lopsided routing can reach; these widths put every
ceiling out of reach. Measured cost of the wider word at a 4096-token chunk:
both stages grow by 238 KiB on a 32-expert/4096-hidden slice and 510 KiB on a
48-expert/6144-hidden one, on the LoRA backend's shared (whole-server)
workspace.
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
