"""The activation vocabulary shared by the MoE LoRA plan and kernel layers.

Its own module because both ends need it and neither may depend on the other:
nothing under ``base_gemm_provider`` imports ``execution_plan``, so the enum
cannot live there without inverting that and pulling pydantic into every
Triton module. Nothing is imported here for the same reason.

TWO AXES, and neither is inferred from the other:

* the POINTWISE function -- this enum;
* GATING -- whether the gate/up buffer is one slice or two, a property of the
  RESIDENT WEIGHT SHAPE, read as ``gate_up_slices`` and passed to the kernels
  as ``NUM_SLICES``.

Every combination is implemented: the S3 kernel takes the function and the
slice count as independent compile-time constants. Collapsing them into one
name is what made non-gated SiLU and gated ReLU2 unservable.
"""

from __future__ import annotations

from enum import Enum


class ActivationFn(str, Enum):
    """The POINTWISE function only. Never the gating.

    Each value IS the string the providers and Triton kernels take, so
    nothing translates between the plan layer and the kernel ABI, and the
    members ARE the accepted set -- ``name in ActivationFn`` is the
    membership test. Adding one is a member here plus a branch in
    ``masked_activation.apply_activation``.
    """

    SILU = "silu"
    RELU2 = "relu2"

    @classmethod
    def parse(cls, name: str) -> ActivationFn:
        """Convert a layer's activation string, or fail closed naming the set.

        Also the validator for callers that only need the check.
        """
        try:
            return cls(name)
        except ValueError:
            raise ValueError(
                f"activation {name!r} is not one of {tuple(fn.value for fn in cls)}"
            ) from None
