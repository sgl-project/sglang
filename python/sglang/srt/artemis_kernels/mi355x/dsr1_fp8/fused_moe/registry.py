"""Registry of exact fused-MoE shapes qualified by Artemis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedMoeKernelSpec:
    """One fail-closed fused-MoE specialization."""

    name: str
    tokens: int
    hidden_size: int = 7168
    intermediate_size_per_partition: int = 256
    experts: int = 257
    top_k: int = 9
    block_size: int = 128
    architecture: str = "gfx950"

    @property
    def hidden_shape(self) -> tuple[int, int]:
        return (self.tokens, self.hidden_size)

    @property
    def w13_shape(self) -> tuple[int, int, int]:
        return (self.experts, 2 * self.intermediate_size_per_partition, self.hidden_size)

    @property
    def w2_shape(self) -> tuple[int, int, int]:
        return (self.experts, self.hidden_size, self.intermediate_size_per_partition)

    @property
    def topk_shape(self) -> tuple[int, int]:
        return (self.tokens, self.top_k)

    @property
    def w13_scale_shape(self) -> tuple[int, int, int]:
        return (
            self.experts,
            2 * self.intermediate_size_per_partition // self.block_size,
            self.hidden_size // self.block_size,
        )

    @property
    def w2_scale_shape(self) -> tuple[int, int, int]:
        return (
            self.experts,
            self.hidden_size // self.block_size,
            self.intermediate_size_per_partition // self.block_size,
        )


FUSED_MOE_KERNELS = (
    FusedMoeKernelSpec(name="dsr1_fp8_tp8_m32", tokens=32),
)

_KERNELS_BY_SHAPE = {
    (
        spec.tokens,
        spec.hidden_size,
        spec.intermediate_size_per_partition,
        spec.experts,
        spec.top_k,
    ): spec
    for spec in FUSED_MOE_KERNELS
}


def find_kernel_spec(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    experts: int,
    top_k: int,
) -> FusedMoeKernelSpec | None:
    """Return the exact registered shape, or ``None`` for stock fallback."""
    return _KERNELS_BY_SHAPE.get(
        (tokens, hidden_size, intermediate_size_per_partition, experts, top_k)
    )


__all__ = ["FUSED_MOE_KERNELS", "FusedMoeKernelSpec", "find_kernel_spec"]
