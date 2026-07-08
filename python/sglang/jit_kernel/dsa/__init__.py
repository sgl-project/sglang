from sglang.srt.utils import is_hip, is_xpu

from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)

if not is_hip() and not is_xpu():
    # CuteDSL uses NVIDIA CUDA DSL which is not available on ROCm or XPU.
    from .cutedsl_paged_mqa_logits import CuteDSLPagedMQALogitsRunner, pick_dsl_expand
else:
    CuteDSLPagedMQALogitsRunner = None
    pick_dsl_expand = None

__all__ = [
    "CuteDSLPagedMQALogitsRunner",
    "pick_dsl_expand",
    "aiter_paged_mqa_logits",
    "cutedsl_paged_mqa_logits",
    "deepgemm_paged_mqa_logits_native",
    "deepgemm_paged_mqa_logits_split",
]
