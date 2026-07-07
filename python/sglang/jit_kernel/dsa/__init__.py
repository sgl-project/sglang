from sglang.srt.utils import is_hip

from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)

if not is_hip():
    # Preserve the original eager import behavior on non-ROCm platforms.
    from .cutedsl_paged_mqa_logits import CuteDSLPagedMQALogitsRunner, pick_dsl_expand

__all__ = [
    "CuteDSLPagedMQALogitsRunner",
    "pick_dsl_expand",
    "aiter_paged_mqa_logits",
    "cutedsl_paged_mqa_logits",
    "deepgemm_paged_mqa_logits_native",
    "deepgemm_paged_mqa_logits_split",
]


if is_hip():

    def __getattr__(name):
        if name in ("CuteDSLPagedMQALogitsRunner", "pick_dsl_expand"):
            from .cutedsl_paged_mqa_logits import (
                CuteDSLPagedMQALogitsRunner,
                pick_dsl_expand,
            )

            globals()["CuteDSLPagedMQALogitsRunner"] = CuteDSLPagedMQALogitsRunner
            globals()["pick_dsl_expand"] = pick_dsl_expand
            return globals()[name]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
