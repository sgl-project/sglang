# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/models/registry.py

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import AbstractSet, Dict, List, Optional, Tuple, Type, Union

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.models.keys()

    def _raise_for_unsupported(self, architectures: List[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_model_cls(self, model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in self.models:
            return None

        return self.models[model_arch]

    def _normalize_archs(
        self,
        architectures: Union[str, List[str]],
    ) -> List[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        return architectures

    def resolve_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> Tuple[Type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)


@lru_cache()
def import_model_classes():
    model_arch_name_to_cls = {}
    package_name = "sglang.srt.models"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}. " f"{e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        assert (
                            tmp.__name__ not in model_arch_name_to_cls
                        ), f"Duplicated model implementation for {tmp.__name__}"
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert (
                        entry.__name__ not in model_arch_name_to_cls
                    ), f"Duplicated model implementation for {entry.__name__}"
                    model_arch_name_to_cls[entry.__name__] = entry

    return model_arch_name_to_cls


ModelRegistry = _ModelRegistry(import_model_classes())
