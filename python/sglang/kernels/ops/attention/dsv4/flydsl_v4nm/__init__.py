# TODO(aiter#5882): remove once ROCm/aiter#5882 is in the aiter pin
"""In-tree copy of ROCm/aiter#5882: DeepSeek-V4 fp8 MLA decode (v4 nm contract)
in one FlyDSL launch on gfx950, for aiter pins that do not ship it yet.

Vendored from ROCm/aiter#5882 at commit 03359c43df (the head adapted to
ROCm/aiter#6377); see the per-file headers. The entry points match the ones
the PR adds to ``aiter.ops.flydsl``. ``unified_kv_kernels.runtime`` prefers
aiter's own copy and only falls back to this one (SGLANG_DSV4_FLYDSL_DECODE_SOURCE).

Needs an aiter with ROCm/aiter#6377 (typed ``_to_ptr_global``): importing this
package raises ImportError on an older aiter, so callers fall back to asm.
"""

import inspect

# AITER owns the FlyDSL toolchain bootstrap and the shared tensor/buffer shims.
# Import it before the local kernel modules so its FlyDSL setup is active.
import aiter as _aiter  # noqa: F401
from aiter.ops.flydsl.kernels import communication_ops_utils as _cu

if "dtype" not in inspect.signature(_cu._to_ptr_global).parameters:
    raise ImportError(
        "the in-tree FlyDSL v4 nm decode (ROCm/aiter#5882) needs an aiter with "
        "ROCm/aiter#6377 (typed communication_ops_utils._to_ptr_global)"
    )

from .mla_v4_decode import (  # noqa: E402
    flydsl_mla_decode_fwd_v4_nm,
    flydsl_mla_v4_decode_fused,
    flydsl_mla_v4_decode_supported,
    flydsl_mla_v4_grouped_pays,
)
from .mla_v4_grouped import flydsl_mla_v4_grouped_verify_meta  # noqa: E402

__all__ = [
    "flydsl_mla_decode_fwd_v4_nm",
    "flydsl_mla_v4_decode_fused",
    "flydsl_mla_v4_decode_supported",
    "flydsl_mla_v4_grouped_pays",
    "flydsl_mla_v4_grouped_verify_meta",
]
