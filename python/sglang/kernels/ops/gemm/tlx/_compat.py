"""Stand-ins for the fbtriton pieces the vendored kernels import or read."""

# fbtriton-only instrumentation knobs that mm_gfx950 reads to skip its compiled-cache
# fast launch; stock Triton has no such instrumentation, so they are off.
_FBTRITON_ONLY_KNOBS = (
    ("runtime", "sanitize_overflow"),
    ("compilation", "fpsan_homomorphic_casts"),
)


class InvalidInput(ValueError):
    """Raised for inputs a kernel does not support (``tlx.ops._catalog.InvalidInput``)."""


def add_missing_triton_knobs() -> None:
    """Define the fbtriton-only knobs as False on stock Triton; a no-op on fbtriton."""
    import triton

    for group, name in _FBTRITON_ONLY_KNOBS:
        knob_group = getattr(triton.knobs, group)
        if not hasattr(knob_group, name):
            # On the class: knobs.scope() clears and restores instance dicts.
            setattr(type(knob_group), name, False)
