"""SynthID hash constants and kernel limits, importable without Triton."""

HASH_MULTIPLIER = 6364136223846793005
HASH_INCREMENT = 1
HASH_ROUNDS = 12
HASH_SHIFT = 5
G_VALUE_SHIFT = 30

# One program holds at most this many candidates: the one-block tournament and
# the fused sample; the wide tournament streams candidates in tiles of it.
MAX_BLOCK_CANDIDATES = 1024
# The widest tournament the kernel tests cover.
MAX_NUM_CANDIDATES = 65536
# The wide tournament packs one g-value bit per key into an int64.
MAX_KEYS = 64
# The sample and projection kernels load a slot's whole history in one block.
MAX_CONTEXT_HISTORY_SIZE = 2048
# The kernels unroll the context hash: ``ngram_len - 1`` tokens.
MAX_CONTEXT_WIDTH = 16
# The projection kernel unrolls the chain; the commit kernel unrolls its tokens.
MAX_CHAIN_WIDTH = 8
