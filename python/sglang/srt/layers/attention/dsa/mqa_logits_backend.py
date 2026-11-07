from enum import Enum
from typing import Callable

from sglang.srt.utils import is_gfx95_supported, is_hip


class DSAMQALogitsBackend(Enum):
    TRITON = "triton"
    FLYDSL = "flydsl"

    @classmethod
    def resolve(cls, value: str) -> "DSAMQALogitsBackend":
        if value in ("auto", "triton"):
            return cls.TRITON
        if value != "flydsl":
            raise ValueError(f"Unknown dsa_mqa_logits_backend: {value!r}")
        if not is_hip() or not is_gfx95_supported():
            raise ValueError("dsa_mqa_logits_backend='flydsl' requires ROCm gfx950.")
        return cls.FLYDSL

    def get_hip_kernel(self) -> Callable:
        if self == DSAMQALogitsBackend.FLYDSL:
            from aiter.ops.flydsl import flydsl_fp8_mqa_logits

            return flydsl_fp8_mqa_logits

        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

        return fp8_mqa_logits
