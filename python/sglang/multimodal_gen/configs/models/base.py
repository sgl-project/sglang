# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, Generic, TypeVar

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

ArchConfigT = TypeVar("ArchConfigT", bound="ArchConfig")


# 1. ArchConfig contains all fields from diffuser's/transformer's config.json (i.e. all fields related to the architecture of the model)
# 2. ArchConfig should be inherited & overridden by each model arch_config
# 3. Any field in ArchConfig is fixed upon initialization, and should be hidden away from users
@dataclass
class ArchConfig:
    extra_attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.refresh_derived_fields()

    def refresh_derived_fields(self) -> None:
        pass

    def __getattr__(self, name: str):
        d = object.__getattribute__(self, "__dict__")
        extras = d.get("extra_attrs")
        if extras is not None and name in extras:
            return extras[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, key, value):
        if key in type(self).__dataclass_fields__:
            object.__setattr__(self, key, value)
        else:
            d = object.__getattribute__(self, "__dict__")
            extras = d.get("extra_attrs")
            if extras is None:
                extras = {}
                d["extra_attrs"] = extras
            extras[key] = value


@dataclass
class ModelConfig(Generic[ArchConfigT]):
    # Every model config parameter can be categorized into either ArchConfig or everything else
    # Diffuser/Transformer parameters
    arch_config: ArchConfigT = field(default_factory=ArchConfig)  # type: ignore[arg-type]

    # sglang-diffusion-specific parameters here
    # i.e. STA, quantization, teacache
    _internal_config_fields: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        self.refresh_derived_fields()

    def __getattr__(self, name):
        # Only called if 'name' is not found in ModelConfig directly
        if hasattr(self.arch_config, name):
            return getattr(self.arch_config, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __getstate__(self):
        # Return a dictionary of attributes to pickle
        # Convert to dict and exclude any problematic attributes
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes from the unpickled state
        self.__dict__.update(state)

    def refresh_model_config(self) -> None:
        pass

    def refresh_derived_fields(self) -> None:
        self.arch_config.refresh_derived_fields()
        self.refresh_model_config()

    @classmethod
    def internal_config_fields(cls) -> set[str]:
        internal_fields: set[str] = set()
        for base in reversed(cls.__mro__):
            internal_fields.update(getattr(base, "_internal_config_fields", ()))
        return internal_fields

    def to_user_dict(self) -> dict[str, Any]:
        state = dataclasses.asdict(self)
        state.pop("arch_config", None)
        for field_name in self.internal_config_fields():
            state.pop(field_name, None)
        return state

    # This should be used only when loading from transformers/diffusers
    def update_model_arch(self, source_model_dict: dict[str, Any]) -> None:
        """
        Update arch_config with source_model_dict
        """
        arch_config = self.arch_config

        for key, value in source_model_dict.items():
            setattr(arch_config, key, value)

        legacy_post_init = type(arch_config).__dict__.get("__post_init__")
        if (
            legacy_post_init is not None
            and legacy_post_init is not ArchConfig.__post_init__
        ):
            legacy_post_init(arch_config)
        else:
            arch_config.refresh_derived_fields()
        self.refresh_model_config()

    def update_model_config(self, source_model_dict: dict[str, Any]) -> None:
        assert (
            "arch_config" not in source_model_dict
        ), "Source model config shouldn't contain arch_config."

        valid_fields = {
            f.name
            for f in fields(self)
            if f.name != "arch_config" and f.name not in self.internal_config_fields()
        }

        for key, value in source_model_dict.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                logger.warning(
                    "%s does not contain field '%s'!", type(self).__name__, key
                )
                raise AttributeError(f"Invalid field: {key}")

        self.refresh_derived_fields()
