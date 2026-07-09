from sglang.srt.utils import is_cuda

from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)

# The CuTe DSL runner requires `cutlass`, which is CUDA-only. Import it eagerly
# only on CUDA; other platforms (ROCm, XPU, CPU) do not import it.
if is_cuda():
    from .cutedsl_paged_mqa_logits import CuteDSLPagedMQALogitsRunner, pick_dsl_expand

__all__ = [
    "CuteDSLPagedMQALogitsRunner",
    "pick_dsl_expand",
    "aiter_paged_mqa_logits",
    "cutedsl_paged_mqa_logits",
    "deepgemm_paged_mqa_logits_native",
    "deepgemm_paged_mqa_logits_split",
]
