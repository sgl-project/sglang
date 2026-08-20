"""The activation vocabulary shared by the MoE LoRA plan and kernel layers.

Its own module, importing nothing, because both ends need it and neither may
depend on the other: living under ``execution_plan`` would pull pydantic into
every Triton module.

The pointwise function (this enum) and gating (``gate_up_slices``, a property
of the resident weight shape) are independent axes -- the S3 kernel takes both
as separate compile-time constants. Collapsing them into one name is what made
non-gated SiLU and gated ReLU2 unservable.
"""

from __future__ import annotations

from enum import Enum


class ActivationFn(str, Enum):
    """The pointwise function only, never the gating.

    Each value IS the string the providers and Triton kernels take, so nothing
    translates between the plan layer and the kernel ABI. Adding one is a
    member here plus a branch in ``masked_activation.apply_activation``.
    """

    SILU = "silu"
    RELU2 = "relu2"

    @classmethod
    def parse(cls, name: str) -> ActivationFn:
        try:
            return cls(name)
        except ValueError:
            raise ValueError(
                f"activation {name!r} is not one of {tuple(fn.value for fn in cls)}"
            ) from None
