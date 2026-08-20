# Port of the FlashInfer GDN CP prefill path (SM100/SM103 CP delta rule and
# its routing) from the 0.6.18.dev20260807 nightly; the pinned FlashInfer
# release (0.6.17) only routes CP prefill on SM90/SM120. Unchanged
# dependencies (chunked kernels, delta-rule helpers, tile scheduler) still
# come from the installed FlashInfer.
from sglang.kernels.ops.attention.gdn_cp_prefill.gdn_prefill import (
    chunk_gated_delta_rule,
)

__all__ = ["chunk_gated_delta_rule"]
