"""The per-attention component identity; a leaf module importable from anywhere."""

from __future__ import annotations


class _ComponentTypeMeta(type):
    """Enum-style lookups on the component ids: ``ComponentType["SWA"]`` by
    name, ``len`` and iteration over the ids known so far."""

    def __getitem__(cls, name: str) -> ComponentType:
        return cls._by_name[name]

    def __iter__(cls):
        return iter(list(cls._by_value.values()))

    def __len__(cls) -> int:
        return len(cls._by_value)


class ComponentType(int, metaclass=_ComponentTypeMeta):
    """A prefix-tree component id, an int so that per-node list storage can be
    indexed directly.

    The fixed ids are the token component (FULL), the first window (SWA), the
    recurrent state (MAMBA) and the C128 sidecar. A tree whose storage keeps
    further windows names each by its sub-pool (`of`), which gives it the next
    free id; a name keeps its id for the life of the process.
    """

    _by_value: dict[int, ComponentType] = {}
    _by_name: dict[str, ComponentType] = {}

    FULL: ComponentType
    SWA: ComponentType
    MAMBA: ComponentType
    C128: ComponentType

    def __new__(cls, value: int) -> ComponentType:
        try:
            return cls._by_value[int(value)]
        except KeyError:
            raise ValueError(f"{value!r} is not a known ComponentType") from None

    @classmethod
    def _create(cls, value: int, name: str) -> ComponentType:
        member = int.__new__(cls, value)
        member._name = name
        cls._by_value[value] = member
        cls._by_name[name] = member
        return member

    @classmethod
    def of(cls, name: str) -> ComponentType:
        """The id a tree slot's component name goes by, registering a further
        window slot's name the first time it is seen."""
        member = cls._by_name.get(name)
        if member is None:
            member = cls._create(len(cls._by_value), name)
        return member

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> int:
        return int(self)

    def __repr__(self) -> str:
        return f"<ComponentType.{self._name}: {int(self)}>"

    def __str__(self) -> str:  # keep human-readable logging
        return self._name.lower()

    def __format__(self, spec: str) -> str:
        return format(str(self), spec)

    def __reduce__(self):
        return (ComponentType.of, (self._name,))

    def __copy__(self) -> ComponentType:
        return self

    def __deepcopy__(self, memo) -> ComponentType:
        return self

    @property
    def is_full(self) -> bool:
        return self == ComponentType.FULL

    @property
    def is_swa(self) -> bool:
        return self == ComponentType.SWA

    @property
    def is_mamba(self) -> bool:
        return self == ComponentType.MAMBA


ComponentType.FULL = ComponentType._create(0, "FULL")
ComponentType.SWA = ComponentType._create(1, "SWA")
ComponentType.MAMBA = ComponentType._create(2, "MAMBA")
ComponentType.C128 = ComponentType._create(3, "C128")

BASE_COMPONENT_TYPE = ComponentType.FULL
