"""Internal storage backing ``SpeculativeAlgorithm.register``. Plugins
should use that classmethod API; do not import from this module directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

WorkerFactory = Callable[["ServerArgs"], Type]
ServerArgsValidator = Callable[["ServerArgs"], None]


class CustomSpecAlgo:
    """A plugin-registered speculative algorithm. Duck-types
    ``SpeculativeAlgorithm`` enum values (same ``is_*()`` / ``create_worker``
    interface).

    Plugins may subclass this to override any ``is_*()`` / ``supports_*()`` /
    ``create_worker`` method (e.g. to integrate with builtin-specific
    branches like ``if spec_algorithm.is_eagle():`` in scheduler /
    model_runner). Pass the subclass via ``spec_class=...`` at registration.

    Defaults: all ``is_*()`` return ``False`` except ``is_speculative``;
    ``supports_spec_v2`` follows ``supports_overlap``.
    """

    def __init__(
        self,
        name: str,
        factory: WorkerFactory,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
    ):
        self.name = name
        self.factory = factory
        self.supports_overlap = supports_overlap
        self.validate_server_args = validate_server_args

    def __repr__(self) -> str:
        return f"CustomSpecAlgo({self.name!r})"

    def is_none(self) -> bool:
        return False

    def is_speculative(self) -> bool:
        return True

    def is_eagle(self) -> bool:
        return False

    def is_eagle3(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False

    def is_standalone(self) -> bool:
        return False

    def is_ngram(self) -> bool:
        return False

    def supports_spec_v2(self) -> bool:
        return self.supports_overlap

    def create_worker(self, server_args: "ServerArgs") -> Type:
        if not server_args.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"Speculative algorithm {self.name} does not support overlap scheduling."
            )
        return self.factory(server_args)


_REGISTRY: Dict[str, CustomSpecAlgo] = {}

# Builtin enum members + the NEXTN alias; plugins cannot shadow these.
_RESERVED_NAMES = frozenset(
    {"DFLASH", "EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM", "NONE"}
)


def register_algorithm(
    name: str,
    *,
    supports_overlap: bool = False,
    validate_server_args: Optional[ServerArgsValidator] = None,
    spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
) -> Callable[[WorkerFactory], WorkerFactory]:
    """Return a decorator that registers a plugin algorithm under ``name``.

    Pass a ``spec_class`` subclass of ``CustomSpecAlgo`` to override any
    ``is_*()`` / ``supports_*()`` / ``create_worker`` method.
    """
    upper = name.upper()
    if upper in _RESERVED_NAMES:
        raise ValueError(
            f"'{upper}' is a reserved speculative algorithm name; cannot be re-registered."
        )
    if upper in _REGISTRY:
        raise ValueError(f"Speculative algorithm '{upper}' already registered.")

    def decorator(factory: WorkerFactory) -> WorkerFactory:
        _REGISTRY[upper] = spec_class(
            name=upper,
            factory=factory,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
        )
        return factory

    return decorator


def get_spec(name: Optional[str]) -> Optional[CustomSpecAlgo]:
    """Return the registered spec for ``name``, or ``None`` for builtin /
    unknown names."""
    if name is None:
        return None
    return _REGISTRY.get(name.upper())
