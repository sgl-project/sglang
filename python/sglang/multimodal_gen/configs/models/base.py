# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field, fields
from typing import Any, Dict

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class ArchConfig:
    """Static model metadata loaded from a Diffusers/Transformers config.

    This includes architecture fields and checkpoint compatibility mappings.
    Runtime capabilities, backend selection, hardware policies, and deployment
    settings belong to the runtime model or server configuration instead.
    """

    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # mapping from huggingface weight names to custom names
    param_names_mapping: dict[str, str | tuple[str, int, int]] = field(
        default_factory=dict
    )
    extra_attrs: Dict[str, Any] = field(default_factory=dict)

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
class ModelConfig:
    # Static component metadata; this is intentionally separate from runtime
    # capabilities and deployment settings.
    arch_config: ArchConfig = field(default_factory=ArchConfig)

    # sglang-diffusion-specific parameters here
    # i.e. STA, quantization, teacache

    def post_diffusers_config_update(self) -> None:
        """Normalize external configuration before constructing the runtime model."""

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

    # This should be used only when loading from transformers/diffusers
    def update_model_arch(self, source_model_dict: dict[str, Any]) -> None:
        """Load static architecture metadata from a source model config."""
        arch_config = self.arch_config

        for key, value in source_model_dict.items():
            setattr(arch_config, key, value)

        if hasattr(arch_config, "__post_init__"):
            arch_config.__post_init__()

    def update_model_config(self, source_model_dict: dict[str, Any]) -> None:
        assert "arch_config" not in source_model_dict, (
            "Source model config shouldn't contain arch_config."
        )

        valid_fields = {f.name for f in fields(self)}

        for key, value in source_model_dict.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                logger.warning(
                    "%s does not contain field '%s'!", type(self).__name__, key
                )
                raise AttributeError(f"Invalid field: {key}")

        if hasattr(self, "__post_init__"):
            self.__post_init__()
