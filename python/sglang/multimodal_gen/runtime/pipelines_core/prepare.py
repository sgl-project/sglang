# SPDX-License-Identifier: Apache-2.0
"""Opt-in two-stage construction; unmigrated pipelines retain their old path."""

import copy
import json
from pathlib import Path

import msgspec
import torch

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_state import PreparedComponent
from sglang.multimodal_gen.runtime.managers.memory_managers.component_loading_order import (
    ComponentLoadSpec,
    order_component_load_specs,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
    prepare_diffusers_component_path_for_loading,
)
from sglang.multimodal_gen.runtime.weight_cache import policy
from sglang.multimodal_gen.runtime.weight_cache.placement import local_device_index
from sglang.multimodal_gen.runtime.weight_cache.plan import (
    ComponentPlan,
    PipelineExecutionPlan,
)


class PreparedPipeline(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    pipeline_cls: type
    model_path: str
    specs: tuple[ComponentLoadSpec, ...]
    cached_components: tuple[PreparedComponent, ...]
    execution_plan: PipelineExecutionPlan
    binding_id: str
    model_index_json: str
    model_subfolder: str | None

    def __post_init__(self):
        names = self.cached_component_names
        if not names or len(set(names)) != len(names):
            raise ValueError("Prepared cache requires distinct, nonempty components")
        if not set(names).issubset(spec.module_name for spec in self.specs):
            raise ValueError(
                "Prepared cache component is absent from the pipeline load plan"
            )

    @property
    def cached_component_names(self):
        return tuple(component.name for component in self.cached_components)

    def component(self, name):
        return next(
            (
                component
                for component in self.cached_components
                if component.name == name
            ),
            None,
        )

    def apply_config(self, server_args):
        # No config rediscovery. Uncached components keep their ordinary loaders.
        server_args.model_subfolder = self.model_subfolder
        for component in self.cached_components:
            component.apply_config(server_args)

    def materialize(self, server_args, *, loaded_modules=None):
        return self.pipeline_cls(
            self.model_path, server_args, prepared=self, loaded_modules=loaded_modules
        )


def prepare_pipeline(pipeline_cls, server_args, *, required=False):
    """Bind audited pipeline roles to the actual loaders' frozen capabilities.

    Ordinary serving never enters this cache-only preparation path.
    """
    if not current_platform.is_cuda():
        if required:
            raise ValueError("The prepared weight-cache capabilities require CUDA")
        return None
    binding = policy.for_pipeline(pipeline_cls)
    if binding is None:
        if required:
            raise ValueError("No prepared weight-cache binding for this pipeline")
        return None
    if (
        server_args.backend == "diffusers"
        or server_args.disagg_role != "monolithic"
        or (
            (server_args.model_subfolder or server_args.model_variant)
            and not binding.supports_subfolder
        )
    ):
        if required:
            raise ValueError(
                "Unsupported backend/role/subfolder/variant for weight cache"
            )
        return None
    args = copy.deepcopy(server_args)
    args._prepared_pipeline = None
    args._weight_cache_admission = None
    if binding.supports_subfolder:
        model_path, model_index = pipeline_cls.resolve_model_config(
            args.model_path, args
        )
        root = Path(model_path)
    else:
        root = Path(
            maybe_download_model(
                args.model_path, force_diffusers_model=True, revision=args.revision
            )
        )
        model_index = json.loads((root / "model_index.json").read_text())
    names = tuple(pipeline_cls._required_config_modules)
    # The first support rows still exclude dual-transformer/conditional variants.
    # Component capability alone must never widen pipeline admission.
    if (
        model_index.get("_class_name") != binding.pipeline_name
        or model_index.get("boundary_ratio") is not None
        or "transformer_2" in model_index
        or any(not model_index.get(name) for name in names)
    ):
        if required:
            raise ValueError(
                "Weight cache requires an audited pipeline component layout"
            )
        return None
    paths = {
        name: (
            str(
                prepare_diffusers_component_path_for_loading(args.component_paths[name])
            )
            if name in args.component_paths
            else str(root / name)
        )
        for name in names
    }
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
    pipeline_cls._validate_direct_gpu_component_selection(model_index, args)
    cached = []
    try:
        if binding.validate_model_index is not None:
            binding.validate_model_index(model_index)
        for selected in binding.components:
            spec = next(spec for spec in specs if spec.module_name == selected.name)
            override = pipeline_cls.component_loaders.get(spec.module_name)
            if override is not None and override is not selected.loader_cls:
                raise ValueError(
                    f"Weight cache has no binding for custom {spec.module_name} loader"
                )
            loader = ComponentLoader.for_component_type(
                spec.load_module_name,
                spec.transformers_or_diffusers,
                spec.architecture,
                loader_cls=override,
                discover_loaders=False,
            )
            if type(loader) is not selected.loader_cls:
                raise ValueError(
                    f"Weight cache has no binding for custom {spec.module_name} loader"
                )
            component = loader.prepare_weight_cache(
                spec,
                args,
                planned_device=(
                    torch.device("cuda", local_device_index(args))
                    if args.weight_cache_mode != "off"
                    else None
                ),
            )
            if component.contract.contract_id != selected.contract_id:
                raise ValueError(
                    "Resolved state contract differs from audited pipeline binding"
                )
            cached.append(component)
    except (ValueError, TypeError, AssertionError):
        if required:
            raise
        return None
    by_name = {component.name: component for component in cached}
    components = []
    for spec in specs:
        backend, _ = args.resolve_component_attention_backend(
            spec.module_name, spec.load_module_name
        )
        component = by_name.get(spec.module_name)
        components.append(
            ComponentPlan(
                spec.module_name,
                spec.transformers_or_diffusers,
                spec.architecture,
                spec.component_model_path,
                args.residency_mode(spec.module_name),
                str(component.attention_backend)
                if component
                else (str(backend) if backend else None),
                f"{component.loader_cls.__module__}.{component.loader_cls.__qualname__}"
                if component
                else "PipelineComponentLoader",
            )
        )
    return PreparedPipeline(
        pipeline_cls,
        str(root),
        specs,
        tuple(cached),
        PipelineExecutionPlan(pipeline_cls.__name__, tuple(components)),
        binding.binding_id,
        json.dumps(model_index, sort_keys=True, separators=(",", ":")),
        args.model_subfolder,
    )
