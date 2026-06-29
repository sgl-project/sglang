# [Deprecated] Re-export shim for backward compatibility. Use dsa.triton_kernel instead.
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.triton_kernel is deprecated; "
    "use sglang.srt.layers.attention.dsa.triton_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.triton_kernel import *  # noqa: F401, F403
