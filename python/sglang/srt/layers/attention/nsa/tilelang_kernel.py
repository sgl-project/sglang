# [Deprecated] Re-export shim for backward compatibility. Use dsa.tilelang_kernel instead.
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.tilelang_kernel is deprecated; "
    "use sglang.kernels.ops.attention.dsa.tilelang_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.tilelang_kernel import *  # noqa: F401, F403
