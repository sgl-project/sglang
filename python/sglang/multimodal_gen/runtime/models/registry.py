# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/registry.py

import ast
import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Set
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NoReturn, TypeVar, cast

import cloudpickle
from torch import nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

MODELS_PATH = os.path.dirname(__file__)
COMPONENT_DIRS = [
    d
    for d in os.listdir(MODELS_PATH)
    if os.path.isdir(os.path.join(MODELS_PATH, d))
    and not d.startswith("__")
    and not d.startswith(".")
]

_IMAGE_ENCODER_MODELS: dict[str, tuple] = {
    # "HunyuanVideoTransformer3DModel": ("image_encoder", "hunyuanvideo", "HunyuanVideoImageEncoder"),
    "CLIPVisionModelWithProjection": ("encoders", "clip", "CLIPVisionModel"),
}


@lru_cache(maxsize=None)
def _discover_and_register_models() -> dict[str, tuple[str, str, str]]:
    discovered_models = _IMAGE_ENCODER_MODELS
    for component in COMPONENT_DIRS:
        component_path = os.path.join(MODELS_PATH, component)
        for filename in os.listdir(component_path):
            if not filename.endswith(".py"):
                continue

            mod_relname = filename[:-3]
            filepath = os.path.join(component_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source, filename=filename)

                entry_class_node = None
                first_class_def = None

                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "EntryClass"
                            ):
                                entry_class_node = node
                                break
                    if first_class_def is None and isinstance(node, ast.ClassDef):
                        first_class_def = node
                if entry_class_node and first_class_def:
                    model_cls_name_list = []
                    value_node = entry_class_node.value

                    # EntryClass = ClassName
                    if isinstance(value_node, ast.Name):
                        model_cls_name_list.append(value_node.id)
                    # EntryClass = ["...", ClassName, ...]
                    elif isinstance(value_node, (ast.List, ast.Tuple)):
                        for elt in value_node.elts:
                            if isinstance(elt, ast.Constant):
                                model_cls_name_list.append(elt.value)
                            elif isinstance(elt, ast.Name):
                                model_cls_name_list.append(elt.id)

                    if model_cls_name_list:
                        for model_cls_str in model_cls_name_list:
                            if model_cls_str in discovered_models:
                                logger.warning(
                                    f"Duplicate architecture found: {model_cls_str}. It will be overwritten."
                                )
                            model_arch = model_cls_str
                            discovered_models[model_arch] = (
                                component,
                                mod_relname,
                                model_cls_str,
                            )

            except Exception as e:
                logger.warning(f"Could not parse {filepath} to find models: {e}")

    return discovered_models


_SGL_DIFFUSION_MODELS = _discover_and_register_models()

_SUBPROCESS_COMMAND = [
    sys.executable,
    "-m",
    "sglang.multimodal_gen.runtime.models.dits.registry",
]

_T = TypeVar("_T")


@dataclass(frozen=True)
class _ModelInfo:
    architecture: str

    @staticmethod
    def from_model_cls(model: type[nn.Module]) -> "_ModelInfo":
        return _ModelInfo(
            architecture=model.__name__,
        )


class _BaseRegisteredModel(ABC):

    @abstractmethod
    def inspect_model_cls(self) -> _ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def load_model_cls(self) -> type[nn.Module]:
        raise NotImplementedError


@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has already been imported in the main process.
    """

    interfaces: _ModelInfo
    model_cls: type[nn.Module]

    @staticmethod
    def from_model_cls(model_cls: type[nn.Module]):
        return _RegisteredModel(
            interfaces=_ModelInfo.from_model_cls(model_cls),
            model_cls=model_cls,
        )

    def inspect_model_cls(self) -> _ModelInfo:
        return self.interfaces

    def load_model_cls(self) -> type[nn.Module]:
        return self.model_cls


def _run_in_subprocess(fn: Callable[[], _T]) -> _T:
    # NOTE: We use a temporary directory instead of a temporary file to avoid
    # issues like https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
    with tempfile.TemporaryDirectory() as tempdir:
        output_filepath = os.path.join(tempdir, "registry_output.tmp")

        # `cloudpickle` allows pickling lambda functions directly
        input_bytes = cloudpickle.dumps((fn, output_filepath))

        # cannot use `sys.executable __file__` here because the script
        # contains relative imports
        returned = subprocess.run(
            _SUBPROCESS_COMMAND, input=input_bytes, capture_output=True
        )

        # check if the subprocess is successful
        try:
            returned.check_returncode()
        except Exception as e:
            # wrap raised exception to provide more information
            raise RuntimeError(
                f"Error raised in subprocess:\n" f"{returned.stderr.decode()}"
            ) from e

        with open(output_filepath, "rb") as f:
            return cast(_T, pickle.load(f))


@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has not been imported in the main process.
    """

    module_name: str
    component_name: str
    class_name: str

    # Performed in another process to avoid initializing CUDA
    def inspect_model_cls(self) -> _ModelInfo:
        return _run_in_subprocess(
            lambda: _ModelInfo.from_model_cls(self.load_model_cls())
        )

    def load_model_cls(self) -> type[nn.Module]:
        mod = importlib.import_module(self.module_name)
        return cast(type[nn.Module], getattr(mod, self.class_name))


@lru_cache(maxsize=128)
def _try_load_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> type[nn.Module] | None:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    current_platform.verify_model_arch(model_arch)
    try:
        return model.load_model_cls()
    except Exception:
        logger.exception("Ignore import error when loading '%s'", model_arch)
        return None


@lru_cache(maxsize=128)
def _try_inspect_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> _ModelInfo | None:
    try:
        return model.inspect_model_cls()
    except Exception:
        logger.exception("Error in inspecting model architecture '%s'", model_arch)
        return None


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: dict[str, _BaseRegisteredModel] = field(default_factory=dict)

    def get_supported_archs(self) -> Set[str]:
        return self.models.keys()

    def register_model(
        self,
        model_arch: str,
        model_cls: type[nn.Module] | str,
    ) -> None:
        """
        Register an external model to be used in vLLM.

        :code:`model_cls` can be either:

        - A :class:`torch.nn.Module` class directly referencing the model.
        - A string in the format :code:`<module>:<class>` which can be used to
          lazily import the model. This is useful to avoid initializing CUDA
          when importing the model and thus the related error
          :code:`RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
        """
        if model_arch in self.models:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.",
                model_arch,
                model_cls,
            )

        if isinstance(model_cls, str):
            split_str = model_cls.split(":")
            if len(split_str) != 2:
                msg = "Expected a string in the format `<module>:<class>`"
                raise ValueError(msg)

            model = _LazyRegisteredModel(*split_str)
        else:
            model = _RegisteredModel.from_model_cls(model_cls)

        self.models[model_arch] = model

    def _raise_for_unsupported(self, architectures: list[str]) -> NoReturn:
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

    def _try_load_model_cls(self, model_arch: str) -> type[nn.Module] | None:
        if model_arch not in self.models:
            return None

        return _try_load_model_cls(model_arch, self.models[model_arch])

    def _try_inspect_model_cls(self, model_arch: str) -> _ModelInfo | None:
        if model_arch not in self.models:
            return None

        return _try_inspect_model_cls(model_arch, self.models[model_arch])

    def _normalize_archs(
        self,
        architectures: str | list[str],
    ) -> list[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        normalized_arch = []
        for model in architectures:
            if model not in self.models:
                raise Exception(
                    f"Unsupported model architecture: {model}. Registered architectures: {architectures}"
                )
                model = "TransformersModel"
            normalized_arch.append(model)
        return normalized_arch

    def inspect_model_cls(
        self,
        architectures: str | list[str],
    ) -> tuple[_ModelInfo, str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return (model_info, arch)

        return self._raise_for_unsupported(architectures)

    def resolve_model_cls(
        self,
        architectures: str | list[str],
    ) -> tuple[type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)


ModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"sglang.multimodal_gen.runtime.models.{component_name}.{mod_relname}",
            component_name=component_name,
            class_name=cls_name,
        )
        for model_arch, (
            component_name,
            mod_relname,
            cls_name,
        ) in _SGL_DIFFUSION_MODELS.items()
    }
)
