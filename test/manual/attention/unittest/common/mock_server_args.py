"""Mock `ServerArgs` factory for attention-backend unit tests.

Production attention backends read many `ServerArgs` attributes and call
several `ServerArgs` methods at backend construction time. The set grows
monotonically: new attention features add new attributes/methods to
`ServerArgs`, and a fixture that mocks `server_args` as a manually-
populated `SimpleNamespace` will silently miss the new field and fail
with `AttributeError` the next time a backend looks it up.

`make_mock_server_args` sidesteps this by instantiating a real
`ServerArgs` (the dataclass) with all defaults from the dataclass
definition, then overlaying the caller's explicit overrides. New
`ServerArgs` attributes are picked up automatically with their default
values; methods like `enable_mamba_extra_buffer()` work because the
object is a real `ServerArgs` instance, so methods are bound correctly.

`__post_init__` is intentionally bypassed (via `object.__new__`) so
fixture callers don't have to supply a real `model_path`; the
validation it performs is irrelevant for module-level attention tests.
"""

import dataclasses

from sglang.srt.server_args import ServerArgs


def make_mock_server_args(**overrides) -> ServerArgs:
    """Return a `ServerArgs` instance with all defaults pre-populated.

    The instance is built by `object.__new__(ServerArgs)` so `__post_init__`
    does not run — fixture callers do not need to supply a valid
    `model_path` or other required-field values.

    Any field with a `default` or `default_factory` in the dataclass
    definition is set automatically. Caller-supplied `overrides` replace
    those defaults; unknown keys are also stored (matching `SimpleNamespace`
    semantics) so fixtures can attach test-only attributes when needed.

    If an override name corresponds to a read-only `@property` on
    `ServerArgs`, the value is stored under `_<name>` instead — many
    `ServerArgs` properties cache through `_<name>` and return it when
    set, so fixture callers can keep using the public name and let this
    helper translate.
    """
    sa = object.__new__(ServerArgs)
    for f in dataclasses.fields(ServerArgs):
        if f.default is not dataclasses.MISSING:
            setattr(sa, f.name, f.default)
        elif f.default_factory is not dataclasses.MISSING:
            setattr(sa, f.name, f.default_factory())
    for k, v in overrides.items():
        cls_attr = getattr(type(sa), k, None)
        if isinstance(cls_attr, property):
            setattr(sa, f"_{k}", v)
        else:
            setattr(sa, k, v)
    return sa
