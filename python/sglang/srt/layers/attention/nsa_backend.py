# [Deprecated] nsa_backend.py is a thin re-export shim for backward compatibility.
# Use dsa_backend.py instead. This file will be removed in a future release.
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa_backend is deprecated; "
    "use sglang.srt.layers.attention.dsa_backend instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa_backend import *  # noqa: F401, F403
from sglang.srt.layers.attention.dsa_backend import (  # noqa: F401
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
    DSAFlashMLAMetadata,
    DSAIndexerMetadata,
    DSAMetadata,
    NativeSparseAttnBackend,
    NativeSparseAttnMultiStepBackend,
    NSAFlashMLAMetadata,
    NSAIndexerMetadata,
    NSAMetadata,
)
