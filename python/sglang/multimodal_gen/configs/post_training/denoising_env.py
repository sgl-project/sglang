# SPDX-License-Identifier: Apache-2.0
"""Which DiT forward args a rollout returns as its denoising env, registered per
pipeline config class.

A config class without a registration inherits the nearest registered base's;
with none anywhere it returns every arg its denoising stage passes, unfiltered.
"""

from typing import Any, Callable, Iterable

# A None entry declares no env args and ends the MRO walk; a missing entry inherits.
_ENV_ARGS_BY_CONFIG: dict[type, frozenset[str] | None] = {}


def register_denoising_env_args(
    args: Iterable[str] | None,
) -> Callable[[type], type]:
    """Register the forward args a pipeline config class returns as its denoising
    env; every other arg is left out. ``None`` registers none, so a subclass with
    another DiT stops inheriting its base's.

    Decorate the class where it is defined, so a process that has the class has its
    registration; above ``@dataclass``, since ``slots=True`` returns a new class.
    """

    def register(config_cls: type) -> type:
        if config_cls in _ENV_ARGS_BY_CONFIG:
            raise ValueError(
                f"{config_cls.__name__} already registered its denoising env args"
            )
        _ENV_ARGS_BY_CONFIG[config_cls] = None if args is None else frozenset(args)
        return config_cls

    return register


def denoising_env_args_of(pipeline_config: Any) -> frozenset[str] | None:
    for config_cls in type(pipeline_config).__mro__:
        if config_cls in _ENV_ARGS_BY_CONFIG:
            return _ENV_ARGS_BY_CONFIG[config_cls]
    return None
