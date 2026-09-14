"""Dependency-free hook shared by pre- and post-publish prefill-buffer sizing."""

from typing import Any, Callable, Optional

_prefill_buffer_ceiling_fn: Optional[Callable[[Any, int], int]] = None


def register_prefill_buffer_ceiling(
    fn: Callable[[Any, int], int],
) -> Callable[[Any, int], int]:
    """Register one provider; repeating the same registration is harmless.

    The provider receives ``(record, default_ceiling)`` and returns the ceiling,
    keeping ``default_ceiling`` for records it does not handle. ``record`` is
    the original argument record, never a view: its fields remain raw inputs.
    Read resolution declarations through ``resolving_view(record)`` from
    ``arg_groups.model_override_base``. The provider must not mutate the record
    and must use the same sizing policy before and after publication.

    Registering a different provider raises instead of replacing the first.
    """
    global _prefill_buffer_ceiling_fn
    if _prefill_buffer_ceiling_fn is not None and _prefill_buffer_ceiling_fn is not fn:
        raise RuntimeError(
            "A different prefill-buffer ceiling provider is already registered"
        )
    _prefill_buffer_ceiling_fn = fn
    return fn


def prefill_buffer_ceiling_of(record: Any, default_ceiling: int) -> int:
    """Return the provider's ceiling, or the default when none is registered."""
    if _prefill_buffer_ceiling_fn is None:
        return default_ceiling
    return _prefill_buffer_ceiling_fn(record, default_ceiling)
