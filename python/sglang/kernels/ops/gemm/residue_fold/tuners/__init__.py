"""FlashInfer-autotuned tactic selection for the residue fold ops.

The tuners share one scaffold (fi_tuner_base): runners[0] is the
pre-existing serving path (never JITs on a cache miss), candidates are
indices into a stable list whose hash goes into the persisted cache key,
and outside the tuning window only precompiled kernels are offered.
"""

from sglang.kernels.ops.gemm.residue_fold.tuners.mext_prefill_tuner import (
    precompile_kloop_sm100,
    tuned_mext_prefill,
)
from sglang.kernels.ops.gemm.residue_fold.tuners.sm10x_fold_tuner import (
    precompile_row_pair,
    tuned_sm10x_fold,
)
from sglang.kernels.ops.gemm.residue_fold.tuners.sm10x_fold_tuner import (
    tuner_enabled as sm10x_tuner_enabled,
)

__all__ = [
    "precompile_kloop_sm100",
    "precompile_row_pair",
    "sm10x_tuner_enabled",
    "tuned_mext_prefill",
    "tuned_sm10x_fold",
]
