# [Deprecated] Re-export shim for backward compatibility. Use dsa.quant_k_cache instead.
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.quant_k_cache is deprecated; "
    "use sglang.kernels.ops.attention.dsa.quant_k_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.quant_k_cache import *  # noqa: F401, F403
