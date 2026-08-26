# SPDX-License-Identifier: Apache-2.0
"""SubBlock -- training-free block-sparse attention routing for video DiTs.

Originally vendored from the standalone SubBlock repository; ``router.py`` and
``kernels.py`` have since diverged from it.

``router.py`` scores every (query block, key block) pair from sub-block-pooled
Q/K and turns the scores into a ``q2k_block_index`` consumed by SGLang's SM90
CuTe-DSL block-sparse FlashAttention or FlashInfer's SM100
``bsa_attn_blk64_fwd`` (bf16, head_dim 128). The estimator and the measurements
behind its defaults are documented there.
"""

from .router import SubBlockRouter, load_bsa_attn_blk64_fwd

__all__ = ["SubBlockRouter", "load_bsa_attn_blk64_fwd"]
