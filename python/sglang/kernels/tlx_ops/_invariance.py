"""Bitwise-invariance request types for `tlx.ops`.

`BitwiseInvarianceSpec` names the logical dimensions that may vary across
calls while overlapping outputs stay bitwise identical; `Space` bundles that
request with the autotune search-space name so `space=` carries both:

    from triton.tlx.ops import BitwiseInvarianceSpec, Space, mm
    spec = BitwiseInvarianceSpec.dims("M")
    c = mm(a, b, space=Space("heuristic", invariance=spec))

Omitted dims are assumed fixed at their first-seen values: the
implementation may specialize to them but must hold the reduction schedule
constant across the invariant dims.

This module is stdlib-only apart from the exception types, so API validation
never needs torch or a GPU.
"""

from __future__ import annotations

import dataclasses

from ._catalog import InvalidInput

__all__ = ["BitwiseInvarianceSpec", "Space", "normalize_space", "gemm_fixed_split_k"]


@dataclasses.dataclass(frozen=True)
class BitwiseInvarianceSpec:
    """Which logical dims may vary while outputs stay bitwise comparable."""

    invariant_dims: frozenset[str] = frozenset()
    all_dims: bool = False  # `all()`: every dim of the consuming op; the wrapper resolves it

    def __post_init__(self) -> None:
        canonical = frozenset(dim.upper() if isinstance(dim, str) else dim for dim in self.invariant_dims)
        object.__setattr__(self, "invariant_dims", canonical)
        if self.all_dims and self.invariant_dims:
            raise InvalidInput("BitwiseInvarianceSpec.all() takes no explicit dims")
        if any(not isinstance(dim, str) for dim in self.invariant_dims):
            raise InvalidInput(f"invariant dims must be strings; got {set(self.invariant_dims)!r}")

    @classmethod
    def dims(cls, *dims: str) -> BitwiseInvarianceSpec:
        """A spec over the named dims (`dims("M")`, `dims("M", "K")`)."""
        return cls(invariant_dims=frozenset(dims))

    @classmethod
    def all(cls) -> BitwiseInvarianceSpec:
        """A spec over every dim of the consuming op, resolved at the wrapper."""
        return cls(all_dims=True)

    def needs_bitwise(self) -> bool:
        """Whether any enforcement is required (the empty spec prunes nothing)."""
        return self.all_dims or bool(self.invariant_dims)

    def invariant_in_dim(self, dim: str) -> bool:
        """Whether `dim` may vary under this spec (kernels see resolved specs)."""
        if self.all_dims:
            raise InvalidInput("BitwiseInvarianceSpec.all() must be resolved against the op vocabulary first")
        return dim.upper() in self.invariant_dims

    def resolve(self, vocab: frozenset[str]) -> BitwiseInvarianceSpec:
        """Expand `all()` against the op vocabulary and reject unknown names."""
        if self.all_dims:
            if not vocab:
                raise InvalidInput("BitwiseInvarianceSpec.all() requires a non-empty op vocabulary; "
                                   "got an empty or unknown vocabulary")
            return BitwiseInvarianceSpec(invariant_dims=frozenset(vocab))
        unknown = set(self.invariant_dims) - set(vocab)
        if unknown:
            raise InvalidInput(f"unknown invariant dims {sorted(unknown)}; vocabulary is {sorted(vocab)}")
        return self


@dataclasses.dataclass(frozen=True)
class Space:
    """An autotune search space plus an optional invariance requirement."""

    name: str  # implementation-defined, e.g. "heuristic" | "full" | "smoke" | "heuristic_2cta"
    invariance: BitwiseInvarianceSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InvalidInput(f"space name must be a non-empty string; got {self.name!r}")
        if self.invariance is not None and not isinstance(self.invariance, BitwiseInvarianceSpec):
            raise InvalidInput(f"space invariance must be a BitwiseInvarianceSpec or None; got {self.invariance!r}")


def normalize_space(space: str | Space | None,
                    default_space: str | None = None) -> tuple[str | None, BitwiseInvarianceSpec]:
    """Split `space=` into (search-space name, invariance request).

    A `None` name means "the implementation default" and is resolved by the
    caller after dispatch. Plain strings carry no invariance request.
    """
    if space is None:
        return default_space, BitwiseInvarianceSpec()
    if isinstance(space, str):
        return space, BitwiseInvarianceSpec()
    if isinstance(space, Space):
        return space.name, space.invariance if space.invariance is not None else BitwiseInvarianceSpec()
    raise InvalidInput(f"space must be a string, Space, or None; got {space!r}")


def gemm_fixed_split_k(spec: BitwiseInvarianceSpec | None) -> int | None:
    """The SPLIT_K a GEMM schedule must use, or None for the normal choice.

    R1/R3 (K invariant): 1 -- the K reduction stays sequential within one
    program per output tile. R2 (M/N invariant, K fixed): 1 until a fixed-K
    specialization exists; uniformity is what matters, and 1 is always
    eligible. R4 (empty or missing spec): None.

    TODO: allow the spec to require a specific SPLIT_K.
    """
    if spec is None or not spec.needs_bitwise():
        return None
    return 1
