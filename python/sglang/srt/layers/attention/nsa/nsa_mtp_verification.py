# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_mtp_verification instead.
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.nsa_mtp_verification is deprecated; "
    "use sglang.srt.layers.attention.dsa.dsa_mtp_verification instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.dsa_mtp_verification import *  # noqa: F401, F403
