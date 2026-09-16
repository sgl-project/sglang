# SPDX-License-Identifier: Apache-2.0
"""Opt-in two-stage construction; unmigrated pipelines retain their old path."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    FrozenTransformerLoad,
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_loading_order import (
    ComponentLoadSpec,
    order_component_load_specs,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
    prepare_diffusers_component_path_for_loading,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.weight_cache import adapters
from sglang.multimodal_gen.runtime.weight_cache.placement import local_device_index
from sglang.multimodal_gen.runtime.weight_cache.plan import (
    ComponentPlan,
    PipelineExecutionPlan,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class PreparedPipeline:
    pipeline_cls: type
    model_path: str
    specs: tuple[ComponentLoadSpec, ...]
    transformer: FrozenTransformerLoad
    execution_plan: PipelineExecutionPlan
    adapter_id: str
    model_index_json: str

    @property
    def adapter(self):
        return adapters.by_id(self.adapter_id)

    def apply_config(self, server_args):
        # The resolver already updated this config. Do not discover or update it
        # again in materialization. Other components keep their ordinary loaders.
        recipe = self.transformer.thaw()
        server_args.model_subfolder = recipe.server_args.model_subfolder
        server_args.pipeline_config.dit_config = recipe.init_params["config"]
        server_args.model_paths["transformer"] = recipe.server_args.model_paths[
            "transformer"
        ]

    def materialize(self, server_args, *, loaded_modules=None):
        return self.pipeline_cls(
            self.model_path, server_args, prepared=self, loaded_modules=loaded_modules
        )


def prepare_pipeline(pipeline_cls, server_args, *, required=False):
    """Resolve admitted DiTs without constructing a pipeline/module.

    This is an explicit preparation context, not an uninitialized fake pipeline
    instance. Ordinary eligible pipelines and cache mode consume the same recipe.
    Other pipeline/configurations remain on their existing ordinary path.
    """
    if not current_platform.is_cuda():
        if required:
            raise ValueError("The prepared weight-cache adapters require CUDA")
        return None
    adapter = adapters.for_pipeline(pipeline_cls)
    if adapter is None:
        if required:
            raise ValueError("No prepared weight-cache adapter for this pipeline")
        return None
    if (
        server_args.backend == "diffusers"
        or server_args.disagg_role != "monolithic"
        or (
            (server_args.model_subfolder or server_args.model_variant)
            and not getattr(adapter, "SUPPORTS_SUBFOLDER", False)
        )
    ):
        if required:
            raise ValueError(
                "Unsupported backend/role/subfolder/variant for weight cache"
            )
        return None
    args = copy.deepcopy(server_args)
    # Do not recursively include an earlier plan in a frozen recipe.
    args._prepared_pipeline = None
    args._weight_cache_admission = None
    if getattr(adapter, "SUPPORTS_SUBFOLDER", False):
        model_path, model_index = pipeline_cls.resolve_model_config(
            args.model_path, args
        )
        root = Path(model_path)
    else:
        root = Path(
            maybe_download_model(
                args.model_path,
                force_diffusers_model=True,
                revision=args.revision,
            )
        )
        model_index = json.loads((root / "model_index.json").read_text())
    names = tuple(pipeline_cls._required_config_modules)
    if (
        model_index.get("_class_name") != adapter.PIPELINE_NAME
        or model_index.get("boundary_ratio") is not None
        or "transformer_2" in model_index
        or any(not model_index.get(name) for name in names)
    ):
        if required:
            raise ValueError(
                "Weight cache requires the admitted single-transformer pipeline"
            )
        return None
    paths = {
        name: (
            str(
                prepare_diffusers_component_path_for_loading(
                    server_args.component_paths[name]
                )
            )
            if name in server_args.component_paths
            else str(root / name)
        )
        for name in names
    }
    config = json.loads((Path(paths["transformer"]) / "config.json").read_text())
    if not adapter.supports_config(config):
        if required:
            raise ValueError(
                f"Weight cache adapter supports {adapter.MODEL_LABEL} only"
            )
        return None
    if (
        pipeline_cls.component_loaders.get("transformer", TransformerLoader)
        is not TransformerLoader
    ):
        if required:
            raise ValueError(
                "Weight cache has no adapter for a custom transformer loader"
            )
        return None
    pipeline_cls._validate_direct_gpu_component_selection(model_index, args)
    loader = TransformerLoader()
    loader.component_load_precision(args, "transformer")
    loader.resolve_component_quantization_override(args, "transformer")
    loader.resolve_component_direct_gpu_loading(args, "transformer")
    transformer_backend, _ = args.resolve_component_attention_backend("transformer")
    attention = str(transformer_backend) if transformer_backend is not None else "fa"
    try:
        if hasattr(adapter, "validate_model_index"):
            adapter.validate_model_index(model_index)
        frozen = loader.prepare_customized(
            paths["transformer"],
            args,
            "transformer",
            planned_device=torch.device("cuda", local_device_index(args)),
        ).freeze()
        adapter.validate_supported(
            frozen, pipeline_name=pipeline_cls.__name__, attention=attention
        )
    except (ValueError, TypeError):
        if required:
            raise
        return None
    specs = tuple(
        order_component_load_specs(
            [
                ComponentLoadSpec(
                    name,
                    name,
                    paths[name],
                    model_index[name][0],
                    model_index[name][1],
                    index,
                )
                for index, name in enumerate(names)
            ]
        )
    )
    components = []
    for spec in specs:
        backend, _ = args.resolve_component_attention_backend(spec.module_name)
        components.append(
            ComponentPlan(
                spec.module_name,
                spec.transformers_or_diffusers,
                spec.architecture,
                spec.component_model_path,
                args.residency_mode(spec.module_name),
                (
                    str(backend)
                    if backend
                    else ("fa" if spec.module_name == "transformer" else None)
                ),
                required and spec.module_name == "transformer",
                (
                    "TransformerLoader.customized"
                    if spec.module_name == "transformer"
                    else "PipelineComponentLoader"
                ),
            )
        )
    return PreparedPipeline(
        pipeline_cls,
        str(root),
        specs,
        frozen,
        PipelineExecutionPlan(pipeline_cls.__name__, tuple(components)),
        adapter.ADAPTER_ID,
        json.dumps(model_index, sort_keys=True, separators=(",", ":")),
    )
