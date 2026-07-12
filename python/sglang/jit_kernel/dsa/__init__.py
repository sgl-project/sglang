from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)


def pick_dsl_expand(*args, **kwargs):
    from .cutedsl_paged_mqa_logits import pick_dsl_expand as _pick_dsl_expand

    return _pick_dsl_expand(*args, **kwargs)


def __getattr__(name: str):
    if name == "CuteDSLPagedMQALogitsRunner":
        from .cutedsl_paged_mqa_logits import CuteDSLPagedMQALogitsRunner

        return CuteDSLPagedMQALogitsRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CuteDSLPagedMQALogitsRunner",
    "pick_dsl_expand",
    "aiter_paged_mqa_logits",
    "cutedsl_paged_mqa_logits",
    "deepgemm_paged_mqa_logits_native",
    "deepgemm_paged_mqa_logits_split",
]
