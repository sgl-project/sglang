# [Deprecated] Re-export shim for backward compatibility. Use dsa.dequant_k_cache instead.
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.dequant_k_cache is deprecated; "
    "use sglang.srt.layers.attention.dsa.dequant_k_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.dequant_k_cache import *  # noqa: F401, F403
