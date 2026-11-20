# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SynthID hash constants and the shapes its Triton kernels support; imports
without Triton. The kernels are in ``synthid_triton``."""

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
