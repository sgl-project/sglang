# SPDX-License-Identifier: Apache-2.0
"""SubBlock -- training-free block-sparse attention routing for video DiTs.

Originally vendored from the standalone SubBlock repository; ``router.py`` and
``kernels.py`` have since diverged from it.

``router.py`` scores every (query block, key block) pair from sub-block-pooled
Q/K and turns the scores into the ``q2k_block_index`` that FlashInfer's
``bsa_attn_blk64_fwd`` consumes (SM100, bf16, head_dim 128). The estimator and
the measurements behind its defaults are documented there.
"""

from .router import SubBlockRouter, load_bsa_attn_blk64_fwd

__all__ = ["SubBlockRouter", "load_bsa_attn_blk64_fwd"]
