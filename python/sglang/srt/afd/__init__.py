"""AFD (Attention-FFN Disaggregation).

Splits attention and MoE expert computation onto separately launched
instances. The attn instance runs the full model except routed experts;
the ffn instance serves expert computation for every MoE layer.

All AFD-specific logic lives in this package; existing files only carry
thin entry-point branches.
"""

from sglang.srt.afd.afd_config import AFDRole, afd_role_from_args

__all__ = ["AFDRole", "afd_role_from_args"]
