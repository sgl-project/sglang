from __future__ import annotations

from enum import Enum

from sglang.srt.utils import is_hip, is_sm100_supported


class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"
    TORCH = "torch"

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER

    def is_torch(self) -> bool:
        return self == DSAPagedMQALogitsBackend.TORCH

    def uses_deepgemm_metadata(self) -> bool:
        return self in (
            DSAPagedMQALogitsBackend.DEEPGEMM,
            DSAPagedMQALogitsBackend.CUTEDSL,
        )

    @staticmethod
    def resolve(value: str) -> DSAPagedMQALogitsBackend:
        if is_hip():
            if value not in ("auto", "aiter"):
                raise ValueError(
                    f"dsa_paged_mqa_logits_backend={value!r} is not supported on "
                    "ROCm; only 'aiter' is implemented."
                )
            return DSAPagedMQALogitsBackend.AITER

        if value == "torch":
            return DSAPagedMQALogitsBackend.TORCH
        if value == "auto" or value == "deepgemm":
            return DSAPagedMQALogitsBackend.DEEPGEMM
        if value == "aiter":
            raise ValueError("dsa_paged_mqa_logits_backend='aiter' requires ROCm.")
        if value == "cutedsl":
            if not is_sm100_supported():
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='cutedsl' requires SM100 (Blackwell)."
                )
            return DSAPagedMQALogitsBackend.CUTEDSL
        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")
