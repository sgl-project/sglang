# [Deprecated] Re-export shim for backward compatibility. Use dsa.transform_index instead.
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.transform_index is deprecated; "
    "use sglang.srt.layers.attention.dsa.transform_index instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.transform_index import *  # noqa: F401, F403
