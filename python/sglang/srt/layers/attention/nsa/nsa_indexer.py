# [Deprecated] Re-export shim for backward compatibility. Use dsa.nsa_indexer instead.
import warnings
warnings.warn(
    "sglang.srt.layers.attention.nsa.nsa_indexer is deprecated; "
    "use sglang.srt.layers.attention.dsa.nsa_indexer instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.nsa_indexer import *  # noqa: F401, F403
