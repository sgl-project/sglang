"""Packed direct-schedule ABI shared by the host builder and device decoder."""

EXPERT_SHIFT = 0
EXPERT_BITS = 10
EXPERT_MASK = (1 << EXPERT_BITS) - 1

TOKEN_CLUSTER_SHIFT = EXPERT_SHIFT + EXPERT_BITS
TOKEN_CLUSTER_BITS = 10
TOKEN_CLUSTER_MASK = (1 << TOKEN_CLUSTER_BITS) - 1

OUTPUT_CLUSTER_SHIFT = TOKEN_CLUSTER_SHIFT + TOKEN_CLUSTER_BITS
OUTPUT_CLUSTER_BITS = 32 - OUTPUT_CLUSTER_SHIFT
OUTPUT_CLUSTER_MASK = (1 << OUTPUT_CLUSTER_BITS) - 1

MAX_EXPERTS = 1 << EXPERT_BITS
MAX_TOKEN_CLUSTERS = 1 << TOKEN_CLUSTER_BITS
# Deliberately narrower than the 12-bit decoder field: keeping bit 31 clear
# makes every packed int32 non-negative for host inspection and debugging.
MAX_OUTPUT_CLUSTERS = 1 << 11
