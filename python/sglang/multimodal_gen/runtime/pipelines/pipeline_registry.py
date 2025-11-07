# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/models/registry.py
# and https://github.com/sgl-project/sglang/blob/v0.4.3/python/sglang/srt/models/registry.py
import dataclasses
import importlib
import pkgutil
from collections.abc import Set
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.server_args import WorkloadType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_PREPROCESS_WORKLOAD_TYPE_TO_PIPELINE_NAME: dict[WorkloadType, str] = {
    WorkloadType.I2V: "PreprocessPipelineI2V",
    WorkloadType.T2V: "PreprocessPipelineT2V",
}


class PipelineType(str, Enum):
    """
    Enumeration for different pipeline types.

    Inherits from str to allow string comparison for backward compatibility.
    """

    BASIC = "basic"
    PREPROCESS = "preprocess"

    @classmethod
    def from_string(cls, value: str) -> "PipelineType":
        """Convert string to PipelineType enum."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid pipeline type: {value}. Must be one of: {', '.join([t.value for t in cls])}"
            ) from None

    @classmethod
    def choices(cls) -> list[str]:
        """Get all available choices as strings."""
        return [pipeline_type.value for pipeline_type in cls]


@dataclass
class _PipelineRegistry:
    # Keyed by pipeline_type -> architecture -> pipeline_name
    # pipelines[pipeline_type][architecture][pipeline_name] = pipeline_cls
    pipelines: dict[str, dict[str, type[ComposedPipelineBase] | None]] = (
        dataclasses.field(default_factory=dict)
    )

    def get_supported_archs(
        self, pipeline_name_in_config: str, pipeline_type: PipelineType
    ) -> Set[str]:
        """Get supported architectures, optionally filtered by pipeline type and workload type."""
        return set(self.pipelines[pipeline_type.value].keys())

    def _load_preprocess_pipeline_cls(
        self, workload_type: WorkloadType
    ) -> type[ComposedPipelineBase] | None:
        pipeline_name = _PREPROCESS_WORKLOAD_TYPE_TO_PIPELINE_NAME[workload_type]

        return self.pipelines[PipelineType.PREPROCESS.value][pipeline_name]

    def _try_load_pipeline_cls(
        self,
        pipeline_name_in_config: str,
        pipeline_type: PipelineType,
        workload_type: WorkloadType,
    ) -> type[ComposedPipelineBase] | type[LoRAPipeline] | None:
        """Try to load a pipeline class for the given architecture, pipeline type, and workload type."""

        if pipeline_type.value not in self.pipelines:
            return None

        try:
            if pipeline_type == PipelineType.PREPROCESS:
                return self._load_preprocess_pipeline_cls(workload_type)
            elif pipeline_type == PipelineType.BASIC:
                return self.pipelines[pipeline_type.value][pipeline_name_in_config]
            else:
                raise ValueError(f"Invalid pipeline type: {pipeline_type.value}")
        except KeyError as e:
            logger.error(
                f"Please check if the ComposedPipeline class has been defined associated with {pipeline_type.value}.{pipeline_name_in_config}"
            )
            raise e
        return None

    def resolve_pipeline_cls(
        self,
        pipeline_name_in_config: str,
        pipeline_type: PipelineType,
        workload_type: WorkloadType,
    ) -> type[ComposedPipelineBase] | type[LoRAPipeline]:
        """Resolve pipeline class based on pipeline name in the config, pipeline type, and workload type."""
        if not pipeline_name_in_config:
            logger.warning("No pipeline architecture is specified")

        pipeline_cls = self._try_load_pipeline_cls(
            pipeline_name_in_config, pipeline_type, workload_type
        )
        if pipeline_cls is not None:
            return pipeline_cls
        supported_archs = self.get_supported_archs(
            pipeline_name_in_config, pipeline_type
        )
        raise ValueError(
            f"Pipeline architecture '{pipeline_name_in_config}' is not supported for pipeline type '{pipeline_type.value}' "
            f"and workload type '{workload_type.value}'. "
            f"Supported architectures: {supported_archs}"
        )


@lru_cache
def import_pipeline_classes(
    pipeline_types: list[PipelineType] | PipelineType | None = None,
) -> dict[str, dict[str, type[ComposedPipelineBase] | None]]:
    """
    Import pipeline classes based on the pipeline type and workload type.

    Args:
        pipeline_types: The pipeline types to load (basic, preprocess).
                      If None, loads all types.

    Returns:
        A three-level nested dictionary:
        {pipeline_type: {architecture_name: {pipeline_name: pipeline_cls}}}
        e.g., {"basic": {"wan": {"WanPipeline": WanPipeline}}}
    """
    type_to_pipeline_dict: dict[str, dict[str, type[ComposedPipelineBase] | None]] = {}
    package_name: str = "sglang.multimodal_gen.runtime.architectures"

    # Determine which pipeline types to scan
    if isinstance(pipeline_types, list):
        pipeline_types_to_scan = [
            pipeline_type.value for pipeline_type in pipeline_types
        ]
    elif isinstance(pipeline_types, PipelineType):
        pipeline_types_to_scan = [pipeline_types.value]
    else:
        pipeline_types_to_scan = [pt.value for pt in PipelineType]

    logger.info("Loading pipelines for types: %s", pipeline_types_to_scan)

    for pipeline_type_str in pipeline_types_to_scan:
        # Try to load from pipeline-type-specific directory first
        pipeline_type_package_name = f"{package_name}.{pipeline_type_str}"
        pipeline_dict: dict[str, type[ComposedPipelineBase] | None] = {}

        try:
            pipeline_type_package = importlib.import_module(pipeline_type_package_name)
            logger.debug("Successfully imported %s", pipeline_type_package_name)

            for _, arch, ispkg in pkgutil.iter_modules(pipeline_type_package.__path__):

                arch_package_name = f"{pipeline_type_package_name}.{arch}"
                if ispkg:
                    arch_package = importlib.import_module(arch_package_name)
                    for _, module_name, ispkg in pkgutil.walk_packages(
                        arch_package.__path__, arch_package_name + "."
                    ):
                        if not ispkg:
                            pipeline_module = importlib.import_module(module_name)
                            if hasattr(pipeline_module, "EntryClass"):
                                entry_cls_list = pipeline_module.EntryClass
                                if not isinstance(entry_cls_list, list):
                                    entry_cls_list = [entry_cls_list]

                                if isinstance(pipeline_module.EntryClass, list):
                                    pipeline_names = [
                                        pipeline.__name__
                                        for pipeline in pipeline_module.EntryClass
                                    ]
                                else:
                                    pipeline_names = [
                                        pipeline_module.EntryClass.__name__
                                    ]

                                for entry_cls, pipeline_name in zip(
                                    entry_cls_list, pipeline_names
                                ):
                                    assert (
                                        pipeline_name not in pipeline_dict
                                    ), f"Duplicated pipeline implementation for {pipeline_name} in {pipeline_type_str}.{arch_package_name}"

                                    assert hasattr(
                                        entry_cls, "pipeline_name"
                                    ), f"{entry_cls}"
                                    pipeline_dict[pipeline_name] = entry_cls

            type_to_pipeline_dict[pipeline_type_str] = pipeline_dict

        except ImportError as e:
            raise ImportError(
                f"Could not import {pipeline_type_package_name} when importing pipeline classes: {e}"
            ) from None

    # Log summary
    total_pipelines = sum(
        len(pipeline_dict) for pipeline_dict in type_to_pipeline_dict.values()
    )
    logger.info(
        "Loaded %d pipeline classes across %d types",
        total_pipelines,
        len(pipeline_types_to_scan),
    )

    return type_to_pipeline_dict


def get_pipeline_registry(
    pipeline_type: PipelineType | str | None = None,
) -> _PipelineRegistry:
    """
    Get a pipeline registry for the specified mode, pipeline type, and workload type.

    Args:
        pipeline_type: Pipeline type to load. If None and mode is provided, will be derived from mode.

    Returns:
        A pipeline registry instance.
    """
    if isinstance(pipeline_type, str):
        pipeline_type = PipelineType.from_string(pipeline_type)

    pipeline_classes = import_pipeline_classes(pipeline_type)
    return _PipelineRegistry(pipeline_classes)
