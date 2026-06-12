"""Internal storage backing ``SpeculativeAlgorithm.register``. Plugins
should use that classmethod API; do not import from this module directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpecInput

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

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def is_speculative(self) -> bool:
        return True

    def is_eagle(self) -> bool:
        return False

    def is_eagle3(self) -> bool:
        return False

    def is_frozen_kv_mtp(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False

    def is_standalone(self) -> bool:
        return False

    def is_ngram(self) -> bool:
        return False

    def supports_target_verify_for_draft(self) -> bool:
        return False

    def has_draft_kv(self) -> bool:
        # Conservative default: the larger KV reserve.
        return True

    def supports_spec_v2(self) -> bool:
        return self.supports_overlap

    def create_worker(self, server_args: ServerArgs) -> Type:
        if not server_args.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"Speculative algorithm {self.name} does not support overlap scheduling."
            )
        return self.factory(server_args)

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # FIXME: Remove this after the forward mode refactor. Target verify is
        # essentially a fixed sequence length prefill/extend with full cuda
        # graph support. We can use it for target verify, or we can use it for
        # other cases which is not target verify but fixed length prefill.
        # Here, we expose this interface to allow the other use cases.
        return num_draft_tokens

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        server_args: ServerArgs,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:
        return None


_REGISTRY: Dict[str, CustomSpecAlgo] = {}

# CLI spellings that are not ``SpeculativeAlgorithm`` members but still resolve
# to a builtin (e.g. NEXTN -> EAGLE). Reserved alongside the enum members so
# plugins cannot shadow them.
_RESERVED_ALIASES = frozenset({"NEXTN"})


def _reserved_names() -> frozenset:
    """Names plugins cannot register under: every ``SpeculativeAlgorithm``
    member plus ``_RESERVED_ALIASES``.

    Derived from the enum (lazily, to avoid a circular import — ``spec_info``
    imports this module) so any new builtin is reserved automatically without
    editing a second list.
    """
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return frozenset(algo.name for algo in SpeculativeAlgorithm) | _RESERVED_ALIASES


def _assert_custom_spec_algo_conforms(spec_class: Type[CustomSpecAlgo]) -> None:
    """Fail fast if ``spec_class`` drifts from the ``SpeculativeAlgorithm``
    duck-typing contract.

    ``from_string`` returns either type and callers dispatch on the shared
    ``is_*()`` / ``supports_*()`` interface without isinstance checks, so every
    such method on the enum must also exist on the registered spec class —
    otherwise a plugin-registered algo hits ``AttributeError`` at a call site
    (this is how ``is_some`` / ``is_frozen_kv_mtp`` silently went missing). New
    predicates are covered automatically; no second list to maintain.

    Called from ``register_algorithm`` rather than at import time because
    ``spec_info`` imports this module, so ``SpeculativeAlgorithm`` does not yet
    exist while this module is loading; at registration time it is fully
    defined.
    """
    # NOTE: use ``vars()`` not ``dir()`` for the enum — ``EnumMeta.__dir__``
    # hides instance methods, so ``dir(SpeculativeAlgorithm)`` would yield an
    # empty interface and turn this guard into a silent no-op.
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    interface = {
        name
        for name in vars(SpeculativeAlgorithm)
        if name.startswith(("is_", "supports_"))
    }
    missing = sorted(interface - set(dir(spec_class)))
    if missing:
        raise TypeError(
            f"{spec_class.__name__} is missing duck-typed methods from "
            f"SpeculativeAlgorithm: {missing}. Add them to {spec_class.__name__} "
            "so plugin-registered algorithms stay dispatchable."
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
    if upper in _reserved_names():
        raise ValueError(
            f"'{upper}' is a reserved speculative algorithm name; cannot be re-registered."
        )
    if upper in _REGISTRY:
        raise ValueError(f"Speculative algorithm '{upper}' already registered.")
    _assert_custom_spec_algo_conforms(spec_class)

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
