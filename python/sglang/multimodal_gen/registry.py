# SPDX-License-Identifier: Apache-2.0
"""
Central registry for multimodal models.

This module provides a centralized registry for multimodal models, including pipelines
and sampling parameters. It allows for easy registration and retrieval of model
information based on model paths or other identifiers.
"""

import dataclasses
import importlib
import os
import pkgutil
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import Backend

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.sensenova_u1 import (
    SENSENOVA_U1_MODEL_IDS,
    is_sensenova_u1_adapter_only_model,
    is_sensenova_u1_model,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.utils.external_model_package import (
    load_external_model_package,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model_index,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# --- Part 1: Pipeline Discovery ---

_PIPELINE_REGISTRY: Dict[str, Type[ComposedPipelineBase]] = {}
_BUILTIN_PIPELINES_DISCOVERED = False

# Registry for pipeline configuration classes (for safetensors files without model_index.json)
# Maps pipeline_class_name -> (PipelineConfig class, SamplingParams class)
_PIPELINE_CONFIG_REGISTRY: Dict[str, Tuple[Type[PipelineConfig], Type[Any]]] = {}


def _discover_and_register_pipelines():
    """
    Automatically discover and register all ComposedPipelineBase subclasses.
    This function scans the 'sglang.multimodal_gen.runtime.pipelines' package,
    finds modules with an 'EntryClass' attribute, and maps the class's 'pipeline_name'
    to the class itself in a global registry.
    """
    global _BUILTIN_PIPELINES_DISCOVERED
    if _BUILTIN_PIPELINES_DISCOVERED:
        return

    package_name = "sglang.multimodal_gen.runtime.pipelines"
    package = importlib.import_module(package_name)
    _BUILTIN_PIPELINES_DISCOVERED = True

    for _, module_name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if not ispkg:
            try:
                pipeline_module = importlib.import_module(module_name)
            except Exception as exc:
                logger.warning(
                    "Skipping pipeline module %s during discovery due to import failure: %s",
                    module_name,
                    exc,
                )
                logger.debug(
                    "Pipeline import failure details for %s", module_name, exc_info=True
                )
                continue
            if hasattr(pipeline_module, "EntryClass"):
                entry_cls = pipeline_module.EntryClass
                entry_cls_list = (
                    [entry_cls] if not isinstance(entry_cls, list) else entry_cls
                )

                for cls in entry_cls_list:
                    if not issubclass(cls, ComposedPipelineBase):
                        continue
                    if cls.pipeline_name in _PIPELINE_REGISTRY:
                        logger.warning(
                            f"Duplicate pipeline name '{cls.pipeline_name}' found. Overwriting."
                        )
                    _PIPELINE_REGISTRY[cls.pipeline_name] = cls

                    # Special handling for ComfyUI Pipelines:
                    # Auto-register config classes if Pipeline class has them defined
                    # since comfyui get model from a single weight file, so we need to register the config classes here
                    if hasattr(cls, "pipeline_config_cls") and hasattr(
                        cls, "sampling_params_cls"
                    ):
                        _PIPELINE_CONFIG_REGISTRY[cls.pipeline_name] = (
                            cls.pipeline_config_cls,
                            cls.sampling_params_cls,
                        )
    logger.debug(
        f"Registering pipelines complete, {len(_PIPELINE_REGISTRY)} pipelines registered"
    )


def _ensure_registry_initialized() -> None:
    _discover_and_register_pipelines()
    load_external_model_package()


def get_pipeline_config_classes(
    pipeline_class_name: str,
) -> Tuple[Type[PipelineConfig], Type[Any]] | None:
    """
    Get the configuration classes for a pipeline.
    """
    # Ensure pipelines are discovered first
    _ensure_registry_initialized()
    return _PIPELINE_CONFIG_REGISTRY.get(pipeline_class_name)


def get_pipeline_class(
    pipeline_class_name: str,
) -> Type[ComposedPipelineBase] | None:
    """Get a registered pipeline class by name."""
    _ensure_registry_initialized()
    return _PIPELINE_REGISTRY.get(pipeline_class_name)


def get_registered_pipeline_names() -> List[str]:
    _ensure_registry_initialized()
    return list(_PIPELINE_REGISTRY)


# --- Part 2: Config Registration ---
@dataclasses.dataclass
class ConfigInfo:
    """Encapsulates all configuration information required to register a
    diffusers model within this framework."""

    sampling_param_cls: Any
    pipeline_config_cls: Type[PipelineConfig]


# The central registry mapping a model name to its configuration information
_CONFIG_REGISTRY: Dict[str, ConfigInfo] = {}

# Mappings from Hugging Face model paths to our internal model names
_MODEL_HF_PATH_TO_NAME: Dict[str, str] = {}

# Detectors to identify model families from paths or class names
_MODEL_NAME_DETECTORS: List[Tuple[str, Callable[[str], bool]]] = []

# native pipelines do not have a diffusers model_index.json. Keep their path
# aliases next to the resolver that consumes them so CLI detection and
# pipeline selection cannot drift apart
KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS: Dict[str, str] = {
    "ming-image-0.1-design": "MingImagePipeline",
    "minimaxai/minimax-h3": "MiniMaxH3Pipeline",
    "minimax/minimax-h3": "MiniMaxH3Pipeline",
    "fastvideo/fastvideo-fasth3-4-step-preview-v1-vsa-datafree": "FastH3Pipeline",
    "openvdn/vdn-minimax-h3": "VDNH3Pipeline",
    "lerobot/pi05": "Pi05Pipeline",
    "pi05": "Pi05Pipeline",
    "pi0.5": "Pi05Pipeline",
    "hunyuan3d": "Hunyuan3D2Pipeline",
    "flux.2-dev-nvfp4": "Flux2NvfpPipeline",
    "fal/ideogram-v4-fast": "Ideogram4FastPipeline",
    "fal/ideogram-v4-instant": "Ideogram4InstantPipeline",
    "comfy-org/ideogram-4": "Ideogram4Nvfp4Pipeline",
}


def register_configs(
    sampling_param_cls: Any,
    pipeline_config_cls: Type[PipelineConfig],
    hf_model_paths: Optional[List[str]] = None,
    model_detectors: Optional[List[Callable[[str], bool]]] = None,
    pipeline_config_registry_entries: Optional[
        Dict[str, Tuple[Type[PipelineConfig], Type[Any]]]
    ] = None,
) -> str:
    """
    Registers configuration classes for a new model family.
    """
    model_id = str(len(_CONFIG_REGISTRY))

    _CONFIG_REGISTRY[model_id] = ConfigInfo(
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
    )
    if hf_model_paths:
        for path in hf_model_paths:
            if path in _MODEL_HF_PATH_TO_NAME:
                logger.warning(
                    f"Model path '{path}' is already mapped to '{_MODEL_HF_PATH_TO_NAME[path]}' and will be overwritten by '{model_id}'."
                )
            _MODEL_HF_PATH_TO_NAME[path] = model_id

    if model_detectors:
        for detector in model_detectors:
            _MODEL_NAME_DETECTORS.append((model_id, detector))

    if pipeline_config_registry_entries:
        for pipeline_name, (pc_cls, sp_cls) in pipeline_config_registry_entries.items():
            _PIPELINE_CONFIG_REGISTRY.setdefault(pipeline_name, (pc_cls, sp_cls))

    return model_id


def register_pipeline(
    pipeline_cls: Type[ComposedPipelineBase],
    *,
    sampling_param_cls: Any,
    pipeline_config_cls: Type[PipelineConfig],
    hf_model_paths: Optional[List[str]] = None,
    model_detectors: Optional[List[Callable[[str], bool]]] = None,
    overwrite: bool = False,
) -> None:
    """Register an out-of-tree native diffusion pipeline and its configs."""
    _discover_and_register_pipelines()
    if not issubclass(pipeline_cls, ComposedPipelineBase):
        raise TypeError("pipeline_cls must inherit from ComposedPipelineBase")
    if not issubclass(pipeline_config_cls, PipelineConfig):
        raise TypeError("pipeline_config_cls must inherit from PipelineConfig")

    pipeline_name = pipeline_cls.pipeline_name
    existing_pipeline = _PIPELINE_REGISTRY.get(pipeline_name)
    if existing_pipeline is not None and not overwrite:
        raise ValueError(
            f"Pipeline '{pipeline_name}' is already registered; pass overwrite=True to replace it"
        )
    for model_path in hf_model_paths or []:
        if model_path in _MODEL_HF_PATH_TO_NAME and not overwrite:
            raise ValueError(f"Model path '{model_path}' is already registered")
        registered_pipeline = KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS.get(
            model_path.lower()
        )
        if registered_pipeline is not None and not overwrite:
            raise ValueError(
                f"Model path '{model_path}' is already registered for pipeline "
                f"'{registered_pipeline}'"
            )

    _PIPELINE_REGISTRY[pipeline_name] = pipeline_cls
    _PIPELINE_CONFIG_REGISTRY[pipeline_name] = (
        pipeline_config_cls,
        sampling_param_cls,
    )
    config_id = register_configs(
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
        hf_model_paths=hf_model_paths,
        model_detectors=None if overwrite else model_detectors,
    )
    if overwrite and model_detectors:
        _MODEL_NAME_DETECTORS[:0] = [
            (config_id, detector) for detector in model_detectors
        ]
    for model_path in hf_model_paths or []:
        KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS[model_path.lower()] = pipeline_name
    _get_config_info.cache_clear()
    get_model_info.cache_clear()
    logger.info(
        "Registered external diffusion pipeline '%s' from %s",
        pipeline_name,
        pipeline_cls.__module__,
    )


_configs_discovered: bool = False

# SANA-WM (register BEFORE generic SANA T2I to prevent "sana" detector false-match)
# SANA-Video (register before generic SANA to avoid detector overlap).
_CONFIG_REGISTER_PRIORITY: Tuple[str, ...] = ("sana_wm", "sana_video")


def _discover_and_register_configs() -> None:
    global _configs_discovered
    if _configs_discovered:
        return
    _configs_discovered = True

    package_name = "sglang.multimodal_gen.configs.pipeline_configs"
    package = importlib.import_module(package_name)

    discovered = []
    for _, module_name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if not ispkg:
            try:
                config_module = importlib.import_module(module_name)
            except Exception as exc:
                logger.warning(
                    f"Skipping config module {module_name} during discovery due to import failure: {exc}",
                )
                continue
            if hasattr(config_module, "register"):
                discovered.append((module_name, config_module))

    def _sort_key(item):
        short_name = item[0].rsplit(".", 1)[-1]
        try:
            return (0, _CONFIG_REGISTER_PRIORITY.index(short_name))
        except ValueError:
            return (1, 0)

    discovered.sort(key=_sort_key)

    for module_name, config_module in discovered:
        try:
            config_module.register()
        except Exception as exc:
            logger.warning(
                f"register() failed for {module_name}: {exc}",
                exc_info=True,
            )


def get_model_short_name(model_id: str) -> str:
    if "/" in model_id:
        return model_id.rstrip("/").split("/")[-1]
    else:
        return model_id


def _normalize_hf_cache_path(path: str) -> str:
    """Normalize a local HuggingFace cache path before substring matching.

    We match registered repo ids like ``org/repo`` against cache fragments like ``models--org--repo`` that appear in snapshot/blob paths.
    """
    return os.path.normpath(path).lower().replace("\\", "/")


def has_registered_diffusion_model_path(model_path: str) -> bool:
    _ensure_registry_initialized()
    all_model_hf_paths = sorted(_MODEL_HF_PATH_TO_NAME.keys(), key=len, reverse=True)

    if is_sensenova_u1_model(model_path):
        return True

    if model_path in _MODEL_HF_PATH_TO_NAME:
        return True

    model_short_name = get_model_short_name(model_path.lower())
    for registered_model_hf_id in all_model_hf_paths:
        if registered_model_hf_id.lower() in SENSENOVA_U1_MODEL_IDS:
            continue
        registered_model_name = get_model_short_name(registered_model_hf_id.lower())
        if registered_model_name in model_short_name:
            return True

    normalized_model_path = _normalize_hf_cache_path(model_path)
    for registered_model_hf_id in all_model_hf_paths:
        if registered_model_hf_id.lower() in SENSENOVA_U1_MODEL_IDS:
            continue
        cache_repo_fragment = (
            f"models--{registered_model_hf_id.lower().replace('/', '--')}"
        )
        if cache_repo_fragment in normalized_model_path:
            return True

    return False


@lru_cache(maxsize=1)
def _get_config_info(
    model_path: str, model_id: Optional[str] = None
) -> Optional[ConfigInfo]:
    """
    Gets the ConfigInfo for a given model path using mappings and detectors.
    """
    _ensure_registry_initialized()
    all_model_hf_paths = sorted(_MODEL_HF_PATH_TO_NAME.keys(), key=len, reverse=True)

    # 0. Explicit model_id override: match by short name
    if model_id is not None:
        model_id_lower = model_id.lower()
        for registered_hf_id in all_model_hf_paths:
            if get_model_short_name(registered_hf_id).lower() == model_id_lower:
                logger.debug(
                    f"Resolved model via explicit --model-id '{model_id}' → '{registered_hf_id}'."
                )
                return _CONFIG_REGISTRY.get(_MODEL_HF_PATH_TO_NAME[registered_hf_id])
        logger.warning(
            f"--model-id '{model_id}' did not match any registered model; "
            "falling back to automatic detection."
        )

    # SenseNova Hub IDs require an exact match, while local checkpoints are
    # identified from their config metadata rather than their directory name.
    if is_sensenova_u1_model(model_path):
        for registered_hf_id in all_model_hf_paths:
            if registered_hf_id.lower() in SENSENOVA_U1_MODEL_IDS:
                return _CONFIG_REGISTRY.get(_MODEL_HF_PATH_TO_NAME[registered_hf_id])

    # 1. Exact match
    if model_path in _MODEL_HF_PATH_TO_NAME:
        model_id = _MODEL_HF_PATH_TO_NAME[model_path]
        logger.debug(f"Resolved model path '{model_path}' from exact path match.")
        return _CONFIG_REGISTRY.get(model_id)

    # 2. Partial match: find the best (longest) match against all registered model hf paths.
    model_short_name = get_model_short_name(model_path.lower())
    for registered_model_hf_id in all_model_hf_paths:
        if registered_model_hf_id.lower() in SENSENOVA_U1_MODEL_IDS:
            continue
        registered_model_name = get_model_short_name(registered_model_hf_id.lower())

        if registered_model_name in model_short_name:
            logger.debug(
                f"Resolved model name '{registered_model_hf_id}' from partial path match."
            )
            model_id = _MODEL_HF_PATH_TO_NAME[registered_model_hf_id]
            return _CONFIG_REGISTRY.get(model_id)

    # 2b. Match local HuggingFace cache snapshot/blob paths such as:
    #   .../models--org--repo/snapshots/<hash>
    # This lets users pass a local HF cache snapshot directory directly even
    # when its basename is only the snapshot hash.
    # Example:
    #    /xxx/models--black-forest-labs--FLUX.2-dev-NVFP4/snapshots/142b87e70bc3006937b7093d89ff287b5f59f071
    # -> models--black-forest-labs--flux.2-dev-nvfp4 (to match with cache_repo_fragment)
    normalized_model_path = _normalize_hf_cache_path(model_path)
    for registered_model_hf_id in all_model_hf_paths:
        if registered_model_hf_id.lower() in SENSENOVA_U1_MODEL_IDS:
            continue
        cache_repo_fragment = (
            f"models--{registered_model_hf_id.lower().replace('/', '--')}"
        )
        if cache_repo_fragment in normalized_model_path:
            logger.debug(
                "Resolved HuggingFace cache path '%s' to registered model '%s'.",
                model_path,
                registered_model_hf_id,
            )
            model_id = _MODEL_HF_PATH_TO_NAME[registered_model_hf_id]
            return _CONFIG_REGISTRY.get(model_id)

    # 3. Use detectors
    config = maybe_download_model_index(model_path)
    pipeline_name = config.get("_class_name", "").lower()

    matched_model_names = []
    for model_id, detector in _MODEL_NAME_DETECTORS:
        if detector(model_path.lower()) or detector(pipeline_name):
            logger.debug(
                f"Matched model name '{model_id}' using a registered detector."
            )
            matched_model_names += [model_id]

    if len(matched_model_names) >= 1:
        if len(matched_model_names) > 1:
            logger.warning(
                "More than one model name is matched, using the first matched"
            )
        model_id = matched_model_names[0]
        return _CONFIG_REGISTRY.get(model_id)
    else:
        logger.debug(
            f"No model info found for model path: {model_path}. "
            f"Please check the model path or specify the model_id explicitly."
        )
        return None


# --- Part 3: Main Resolver ---


@dataclasses.dataclass
class ModelInfo:
    """
    Encapsulates all configuration information required to register a
    diffusers model within this framework.
    """

    pipeline_cls: Type[ComposedPipelineBase]
    sampling_param_cls: Any
    pipeline_config_cls: Type[PipelineConfig]


def _get_diffusers_model_info(
    model_path: Optional[str] = None,
    model_id: Optional[str] = None,
) -> ModelInfo:
    """
    Get model info for diffusers backend.

    Returns a ModelInfo with DiffusersPipeline and generic configs.
    When model_path is provided and has a registered native config,
    inherits task_type from it so that validation (e.g. accepts_image_input)
    works correctly even under the diffusers backend.
    """
    from sglang.multimodal_gen.configs.pipeline_configs.diffusers_generic import (
        DIFFUSERS_TASK_TYPE_TO_CONFIG,
        DiffusersGenericPipelineConfig,
    )
    from sglang.multimodal_gen.configs.sample.diffusers_generic import (
        DiffusersGenericSamplingParams,
    )
    from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import (
        DiffusersPipeline,
    )

    sampling_param_cls = DiffusersGenericSamplingParams
    pipeline_config_cls = DiffusersGenericPipelineConfig

    # If there is a registered native config for this model, inherit its task_type.
    # We use pre-defined static subclasses instead of make_dataclass so the config
    # class is pickle-safe for multiprocessing spawn (fixes #21453).
    if model_path is not None:
        config_info = _get_config_info(model_path, model_id=model_id)
        if config_info is not None:
            sampling_param_cls = config_info.sampling_param_cls
            native_task_type = config_info.pipeline_config_cls.task_type
            if native_task_type != DiffusersGenericPipelineConfig.task_type:
                pipeline_config_cls = DIFFUSERS_TASK_TYPE_TO_CONFIG.get(
                    native_task_type, DiffusersGenericPipelineConfig
                )
                logger.debug(
                    "Inherited task_type=%s from native config for diffusers backend",
                    native_task_type.name,
                )

    return ModelInfo(
        pipeline_cls=DiffusersPipeline,
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
    )


@lru_cache(maxsize=1)
def get_model_info(
    model_path: str,
    backend: Optional[Union[str, "Backend"]] = None,
    model_id: Optional[str] = None,
) -> Optional[ModelInfo]:
    """
    Resolves all necessary classes (pipeline, sampling, config) for a given model path.

    This function serves as the main entry point for model resolution. It performs two main tasks:
    1. Dynamically resolves the pipeline class by reading 'model_index.json' and matching
       '_class_name' against an auto-discovered registry of pipeline implementations.
    2. Resolves the associated configuration classes (for sampling and pipeline) using a
       manually registered mapping based on the model path.

    Args:
        backend: Backend to use ('auto', 'sglang', 'diffusers'). If None, uses 'auto'.

    """
    # import Backend enum here to avoid circular imports
    from sglang.multimodal_gen.runtime.server_args import Backend

    # Normalize backend
    if backend is None:
        backend = Backend.AUTO
    elif isinstance(backend, str):
        backend = Backend.from_string(backend)

    if is_sensenova_u1_adapter_only_model(model_path):
        logger.error(
            "SenseNova-U1 adapter-only checkpoint '%s' does not contain base "
            "model weights or config. SenseNova-U1 adapters are not supported "
            "yet; use the base checkpoint 'sensenova/SenseNova-U1.5-8B-MoT' "
            "directly.",
            model_path,
        )
        return None

    # Handle explicit diffusers backend
    if backend == Backend.DIFFUSERS:
        logger.info(
            "Using diffusers backend for model '%s' (explicitly requested)", model_path
        )
        return _get_diffusers_model_info(model_path=model_path, model_id=model_id)

    # For AUTO or SGLANG backend, try native implementation first
    # 1. Discover all available pipeline classes and cache them
    _ensure_registry_initialized()

    # Detect quantized models and fallback to diffusers
    is_quantized = any(q in model_path.lower() for q in ["-4bit", "-awq", "-gptq"])
    if is_quantized and backend != Backend.DIFFUSERS:
        logger.info(
            "Detected a quantized model format ('%s'). "
            "The native sglang-diffusion engine currently only supports BF16/FP16. "
            "Falling back to diffusers backend.",
            model_path,
        )
        return _get_diffusers_model_info(model_path=model_path, model_id=model_id)

    # 2. Get pipeline class - check non-diffusers models first
    pipeline_class_name = get_non_diffusers_pipeline_name(model_path)
    if pipeline_class_name:
        # Known non-diffusers model, skip model_index.json download
        logger.debug(
            f"Using registered pipeline '{pipeline_class_name}' for non-diffusers model '{model_path}'"
        )
    else:
        # Try to get from model_index.json
        try:
            config = maybe_download_model_index(model_path)
        except Exception as e:
            logger.error(f"Could not read model config for '{model_path}': {e}")
            if backend == Backend.AUTO:
                logger.info("Falling back to diffusers backend")
                return _get_diffusers_model_info(
                    model_path=model_path, model_id=model_id
                )
            return None

        pipeline_class_name = config.get("_class_name")
        if not pipeline_class_name:
            logger.error(
                f"'_class_name' not found in model_index.json for '{model_path}'"
            )
            if backend == Backend.AUTO:
                logger.info("Falling back to diffusers backend")
                return _get_diffusers_model_info(
                    model_path=model_path, model_id=model_id
                )
            return None

    pipeline_cls = _PIPELINE_REGISTRY.get(pipeline_class_name)
    if not pipeline_cls:
        if backend == Backend.AUTO:
            logger.warning(
                f"Pipeline class '{pipeline_class_name}' specified in '{model_path}' has no native sglang support. "
                f"Falling back to diffusers backend."
            )
            return _get_diffusers_model_info(model_path=model_path, model_id=model_id)
        else:
            logger.error(
                f"Pipeline class '{pipeline_class_name}' specified in '{model_path}' is not a registered EntryClass in the framework. "
                f"Available pipelines: {list(_PIPELINE_REGISTRY.keys())}. "
                f"Consider using --backend diffusers to use vanilla diffusers pipeline."
            )
            return None

    # 3. Get configuration classes (sampling, pipeline config)
    config_info = _get_config_info(model_path, model_id=model_id)
    if not config_info:
        if backend == Backend.AUTO:
            logger.warning(
                f"Could not resolve native configuration for model '{model_path}'. "
                f"Falling back to diffusers backend."
            )
            return _get_diffusers_model_info(model_path=model_path, model_id=model_id)
        else:
            logger.error(
                f"Could not resolve configuration for model '{model_path}'. "
                "It is not a registered model path or detected by any registered model family detectors. "
                f"Known model paths: {list(_MODEL_HF_PATH_TO_NAME.keys())}. "
                f"Consider using --backend diffusers to use vanilla diffusers pipeline."
            )
            return None

    # 4. Combine and return the complete model info
    logger.debug("Using native sglang backend for model '%s'", model_path)
    model_info = ModelInfo(
        pipeline_cls=pipeline_cls,
        sampling_param_cls=config_info.sampling_param_cls,
        pipeline_config_cls=config_info.pipeline_config_cls,
    )
    logger.debug(f"Found model info: {model_info}")

    return model_info


_discover_and_register_configs()


def is_known_non_diffusers_multimodal_model(model_path: str) -> bool:
    return get_non_diffusers_pipeline_name(model_path) is not None


def get_non_diffusers_pipeline_name(model_path: str) -> Optional[str]:
    """Get the pipeline name for a known non-diffusers model."""
    if is_sensenova_u1_model(model_path):
        return "SenseNovaU1Pipeline"

    normalized_model_path = _normalize_hf_cache_path(model_path)
    model_short_name = get_model_short_name(normalized_model_path)
    for pattern, pipeline_name in KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS.items():
        pattern = pattern.lower()
        if "/" not in pattern and pattern in normalized_model_path:
            return pipeline_name
        if "/" in pattern and (
            normalized_model_path == pattern
            or model_short_name == get_model_short_name(pattern)
            or f"models--{pattern.replace('/', '--')}" in normalized_model_path
        ):
            return pipeline_name
    return None


def is_registered_diffusion_model_path(model_path: str) -> bool:
    """Return whether the diffusion registry recognizes a model path."""
    return has_registered_diffusion_model_path(model_path) or (
        get_non_diffusers_pipeline_name(model_path) is not None
    )
