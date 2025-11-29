from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import IntEnum, auto
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from sglang.srt.managers.schedule_batch import ModelWorkerBatch

DraftWorkerClass = Callable[..., Any]
DraftWorkerFactory = Callable[..., Any]


class _SpeculativeAlgorithmMeta(type):
    def __iter__(cls) -> Iterator["SpeculativeAlgorithm"]:
        return iter(cls._registration_order)


class SpeculativeAlgorithm(metaclass=_SpeculativeAlgorithmMeta):
    """Registry-backed representation of speculative decoding algorithms."""

    __slots__ = ("name", "value", "_draft_worker_factory")

    _registry_by_name: Dict[str, "SpeculativeAlgorithm"] = {}
    _registry_by_value: Dict[int, "SpeculativeAlgorithm"] = {}
    _registration_order: List["SpeculativeAlgorithm"] = []
    _flags: DefaultDict[str, Set[int]] = defaultdict(set)
    _next_value: int = 0

    def __init__(
        self,
        name: str,
        value: int,
        draft_worker_factory: Optional[DraftWorkerFactory] = None,
    ):
        self.name = name
        self.value = value
        self._draft_worker_factory = draft_worker_factory

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SpeculativeAlgorithm.{self.name}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SpeculativeAlgorithm):
            return self.value == other.value
        return NotImplemented

    def __int__(self) -> int:
        return self.value

    @classmethod
    def register(
        cls,
        name: str,
        *,
        aliases: Optional[Sequence[str]] = None,
        value: Optional[int] = None,
        draft_worker_factory: Optional[DraftWorkerFactory] = None,
    ) -> SpeculativeAlgorithm:
        normalized_name = name.upper()
        if normalized_name in cls._registry_by_name:
            raise ValueError(
                f"SpeculativeAlgorithm '{normalized_name}' already registered"
            )

        if value is None:
            value = cls._next_value
        cls._next_value = max(cls._next_value, value + 1)

        algorithm = cls(
            normalized_name,
            value,
            draft_worker_factory=draft_worker_factory,
        )

        cls._registry_by_name[normalized_name] = algorithm
        cls._registry_by_value[value] = algorithm
        cls._registration_order.append(algorithm)
        setattr(cls, normalized_name, algorithm)

        if aliases:
            cls.register_aliases(algorithm, *aliases)

        return algorithm

    @classmethod
    def register_aliases(cls, algorithm: SpeculativeAlgorithm, *aliases: str) -> None:
        for alias in aliases:
            cls._registry_by_name[alias.upper()] = algorithm

    @classmethod
    def register_draft_worker(
        cls,
        algorithm: SpeculativeAlgorithm | str,
        factory: DraftWorkerFactory,
    ) -> None:
        algo = cls._ensure_algorithm(algorithm)
        algo._draft_worker_factory = factory

    @classmethod
    def _ensure_algorithm(
        cls, algorithm: SpeculativeAlgorithm | str
    ) -> SpeculativeAlgorithm:
        if isinstance(algorithm, SpeculativeAlgorithm):
            return algorithm
        if isinstance(algorithm, str):
            return cls.from_string(algorithm)
        raise TypeError(f"Unsupported algorithm identifier: {algorithm!r}")

    @classmethod
    def _add_flag(
        cls, flag: str | Sequence[str], algorithm: SpeculativeAlgorithm | str
    ) -> None:
        algo = cls._ensure_algorithm(algorithm)
        if isinstance(flag, str):
            flag_iter = (flag,)
        else:
            flag_iter = flag
        for flag_name in flag_iter:
            cls._flags[flag_name.upper()].add(algo.value)

    @classmethod
    def from_string(cls, name: Optional[str]) -> SpeculativeAlgorithm:
        if name is None:
            return cls.NONE
        try:
            return cls._registry_by_name[name.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown speculative algorithm '{name}'") from exc

    @classmethod
    def from_value(cls, value: int) -> SpeculativeAlgorithm:
        try:
            return cls._registry_by_value[value]
        except KeyError as exc:
            raise ValueError(f"Unknown speculative algorithm id {value}") from exc

    def _has_flag(self, flag: str) -> bool:
        return self.value in type(self)._flags.get(flag.upper(), set())

    def is_none(self) -> bool:
        return self is SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        return self._has_flag("EAGLE")

    def is_eagle3(self) -> bool:
        return self._has_flag("EAGLE3")

    def is_standalone(self) -> bool:
        return self._has_flag("STANDALONE")

    def is_ngram(self) -> bool:
        return self._has_flag("NGRAM")

    def create_draft_worker(self, **factory_kwargs: Any) -> Any:
        if self._draft_worker_factory is None:
            return None
        return self._draft_worker_factory(self, **factory_kwargs)


# Registry helpers backed by `SpeculativeAlgorithm`.
_LOCK = threading.RLock()
_REGISTERED_WORKERS: Dict[SpeculativeAlgorithm, DraftWorkerClass] = {}
_FLAG_MARKERS: Dict[str, Callable[[Union[SpeculativeAlgorithm, str]], None]] = {
    "EAGLE": lambda algorithm: SpeculativeAlgorithm._add_flag("EAGLE", algorithm),
    "EAGLE3": lambda algorithm: SpeculativeAlgorithm._add_flag("EAGLE3", algorithm),
    "STANDALONE": lambda algorithm: SpeculativeAlgorithm._add_flag(
        "STANDALONE", algorithm
    ),
    "NGRAM": lambda algorithm: SpeculativeAlgorithm._add_flag("NGRAM", algorithm),
}


def _wrap_worker_class(worker_cls: DraftWorkerClass) -> DraftWorkerFactory:
    def _factory(_: SpeculativeAlgorithm, **kwargs: Any) -> Any:
        return worker_cls(**kwargs)

    return _factory


def register_speculative_algorithm(
    name: str,
    worker_cls: DraftWorkerClass,
    *,
    aliases: Optional[Sequence[str]] = None,
    flags: Optional[Iterable[str]] = None,
    value: Optional[int] = None,
    override_worker: bool = False,
) -> SpeculativeAlgorithm:
    """Register a speculative algorithm and the associated draft worker class.

    Example:
        >>> from sglang.srt.speculative.spec_info import register_speculative_algorithm
        >>> register_speculative_algorithm("MY_ALGO", MyDraftWorker, flags=("EAGLE",))
    """

    name_upper = name.upper()
    with _LOCK:
        try:
            algorithm = SpeculativeAlgorithm.from_string(name_upper)
            exists = True
        except ValueError:
            algorithm = SpeculativeAlgorithm.register(
                name_upper,
                aliases=aliases,
                value=value,
            )
            SpeculativeAlgorithm.register_draft_worker(
                algorithm, _wrap_worker_class(worker_cls)
            )
            exists = False

        if exists:
            if aliases:
                SpeculativeAlgorithm.register_aliases(algorithm, *aliases)
            if not override_worker and algorithm in _REGISTERED_WORKERS:
                raise ValueError(
                    f"Worker already registered for {algorithm!r}. "
                    "Pass override_worker=True to replace it."
                )
            SpeculativeAlgorithm.register_draft_worker(
                algorithm, _wrap_worker_class(worker_cls)
            )

        _REGISTERED_WORKERS[algorithm] = worker_cls

        if flags:
            for flag in flags:
                marker = _FLAG_MARKERS.get(flag.upper())
                if marker is None:
                    raise ValueError(f"Unsupported flag '{flag}'")
                marker(algorithm)

        return algorithm


def list_registered_workers() -> Dict[str, DraftWorkerClass]:
    """Return a snapshot of registered speculative worker classes keyed by algorithm name."""
    with _LOCK:
        return {algo.name: cls for algo, cls in _REGISTERED_WORKERS.items()}


def _create_eagle_worker(**kwargs: Any) -> Any:
    enable_overlap = kwargs.pop("enable_overlap", False)
    if enable_overlap:
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

        return EAGLEWorkerV2(**kwargs)

    from sglang.srt.speculative.eagle_worker import EAGLEWorker

    return EAGLEWorker(**kwargs)


def _create_standalone_worker(**kwargs: Any) -> Any:
    from sglang.srt.speculative.standalone_worker import StandaloneWorker

    return StandaloneWorker(**kwargs)


def _create_ngram_worker(**kwargs: Any) -> Any:
    from sglang.srt.speculative.ngram_worker import NGRAMWorker

    return NGRAMWorker(**kwargs)


# Register built-in algorithms.
# Third-party integrations should import `SpeculativeAlgorithm` and either
# call `register_speculative_algorithm` or use the helpers below to attach
# additional draft workers.
SpeculativeAlgorithm.register("NONE")

register_speculative_algorithm(
    "EAGLE",
    aliases=("NEXTN",),
    worker_cls=_create_eagle_worker,
    flags=("EAGLE",),
)

register_speculative_algorithm(
    "EAGLE3",
    worker_cls=_create_eagle_worker,
    flags=("EAGLE", "EAGLE3"),
)

register_speculative_algorithm(
    "STANDALONE",
    worker_cls=_create_standalone_worker,
    flags=("STANDALONE",),
)

register_speculative_algorithm(
    "NGRAM",
    worker_cls=_create_ngram_worker,
    flags=("NGRAM",),
)


class SpecInputType(IntEnum):
    # NOTE: introduce this to distinguish the SpecInput types of multiple algorithms when asserting in attention backends.
    # If all algorithms can share the same datastrucutre of draft_input and verify_input, consider simplify it
    EAGLE_DRAFT = auto()
    EAGLE_VERIFY = auto()
    NGRAM_VERIFY = auto()


class SpecInput(ABC):
    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type

    def is_draft_input(self) -> bool:
        # FIXME: remove this function which is only used for assertion
        # or use another variable name like `draft_input` to substitute `spec_info`
        return self.spec_input_type == SpecInputType.EAGLE_DRAFT

    def is_verify_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        }

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        pass

    def get_spec_adjusted_global_num_tokens(
        self, forward_batch: ModelWorkerBatch
    ) -> Tuple[List[int], List[int]]:
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in forward_batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in forward_batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob
