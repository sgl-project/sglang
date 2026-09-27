# Temp workaround, make layer utils more fine-grained later
from sglang.srt.layers.utils.common import *

# Deprecated re-export kept for external users; in-repo code subclasses
# sglang.kernels.fused_op.BaseFusedOp directly (RFC #29630).
from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
