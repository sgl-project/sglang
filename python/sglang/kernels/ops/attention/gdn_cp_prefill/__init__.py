# Port of the FlashInfer GDN CP prefill path (SM100/SM103 CP delta rule and its
# routing) from flashinfer main at 76704c4; the pinned FlashInfer release
# (0.6.17) only routes CP prefill on SM90/SM120. Unchanged dependencies
# (chunked kernels, delta-rule helpers, tile scheduler) come from the installed
# FlashInfer.
from sglang.kernels.ops.attention.gdn_cp_prefill.gdn_prefill import (
    chunk_gated_delta_rule,
)

__all__ = ["chunk_gated_delta_rule"]
