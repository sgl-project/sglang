# SPDX-License-Identifier: Apache-2.0
"""ComfyUI single-file DiT checkpoints: spec, registry, and load.

ComfyUI ships a DiT as one ``.safetensors`` (or a GGUF override) with no
``model_index.json`` and its own parameter names. A spec supplies what the
shared loader cannot infer: which DiT config to build, how names map onto
SGLang, and how to reshape tensors whose layout differs. Per-model specs live
in the sibling modules of this package.

Everything else -- meta-device init, FSDP sharding, quantization, CPU
offload -- goes through the regular transformer load path.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.minimax_h3_weights import (
    comfy_quant_key_filter,
    inspect_minimax_h3_safetensors,
    resolve_minimax_h3_checkpoint_quantization,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )

logger = init_logger(__name__)

WeightIterator = Iterator[tuple[str, torch.Tensor]]

# ComfyUI mapping entries reuse the param_names_mapping format:
#   source_regex -> (target_template, merge_index, num_params_to_merge)
ParamNamesMapping = dict[str, tuple[str, int | None, int | None]]


@dataclass(frozen=True)
class ComfyUICheckpointSpec:
    """Per-model knowledge needed to load a ComfyUI checkpoint."""

    dit_cls_name: str
    build_dit_config: Callable[[ServerArgs], Any]
    param_names_mapping: ParamNamesMapping = field(default_factory=dict)
    # Reshapes tensors that param_names_mapping cannot express. Receives the raw
    # safetensors iterator plus the built dit config, yields SGLang-shaped pairs.
    convert_weights: Callable[[WeightIterator, Any], WeightIterator] | None = None
    # Set False for checkpoints that legitimately omit parameters the model
    # declares, such as optional biases.
    strict: bool = True
    # Whether to layer param_names_mapping on top of the DiT config's own
    # mapping. Keep it True when the two act on different names (the config
    # rules then finish the job, e.g. merging split QKV back into a fused
    # parameter). Set False when both claim the same source names, since name
    # mapping is applied repeatedly until it reaches a fixed point and the
    # config rules would rewrite names this spec already resolved.
    inherit_config_mapping: bool = True
    # Serialized ComfyUI quantization: per-layer markers keyed by SGLang module
    # prefix ({} for an unquantized file). MiniMax-H3 reads its own when unset.
    quant_markers: Callable[[list[str]], dict[str, dict[str, Any]]] | None = None
    # Drops checkpoint keys that are not DiT weights (e.g. all-in-one files).
    checkpoint_key_filter: Callable[[str], bool] | None = None


# Methods whose no-arg config quantizes BF16 weights after loading.
_ONLINE_QUANTIZATIONS = frozenset({"fp8", "mxfp4", "convrot_int8"})

_SPEC_REGISTRY: dict[str, ComfyUICheckpointSpec] = {}
_SPECS_DISCOVERED = False


def register_comfyui_checkpoint(
    pipeline_name: str, spec: ComfyUICheckpointSpec
) -> None:
    _SPEC_REGISTRY[pipeline_name] = spec


def _discover_checkpoint_specs() -> None:
    global _SPECS_DISCOVERED
    if _SPECS_DISCOVERED:
        return
    _SPECS_DISCOVERED = True
    from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import (  # noqa: F401
        flux,
        minimax_h3,
        qwen_image,
        qwen_image21,
        zimage,
    )


def get_comfyui_checkpoint_spec(pipeline_name: str) -> ComfyUICheckpointSpec | None:
    _discover_checkpoint_specs()
    return _SPEC_REGISTRY.get(pipeline_name)


def get_registered_comfyui_pipeline_names() -> list[str]:
    _discover_checkpoint_specs()
    return sorted(_SPEC_REGISTRY)


def is_comfyui_single_file(model_path: str) -> bool:
    """ComfyUI ships DiTs as one safetensors or GGUF file with no model_index.json."""
    return os.path.isfile(model_path) and model_path.endswith((".safetensors", ".gguf"))


def _gguf_runtime_layout(tensor_meta: dict, mapping: dict) -> tuple[dict, dict]:
    """Key GGUF metadata by runtime names; return it with the fused-shard plan.

    Shards of one fused runtime layer (Z-Image w1/w3 -> w13) keep their own
    entries and are concatenated on load, under the fused name quant lookup uses.
    """
    from sglang.multimodal_gen.runtime.loader.gguf_weights import (
        remap_gguf_tensor_meta,
    )

    map_name = get_param_names_mapping(mapping)
    plain, fused = {}, defaultdict(dict)
    for name, meta in tensor_meta.items():
        target, index, _ = map_name(name)
        if index is None:
            plain[name] = meta
        else:
            fused[target][index] = (name, meta)

    layout = remap_gguf_tensor_meta(plain, lambda name: map_name(name)[0])
    shard_plan = {}
    for target, parts in fused.items():
        metas = [parts[i][1] for i in sorted(parts)]
        first = metas[0]
        if len({(m.ggml_type, m.is_packed, m.logical_shape[1:]) for m in metas}) > 1:
            raise ValueError(f"GGUF shards of {target} differ in type or width")
        param_name = (
            f"{target.removesuffix('.weight')}.qweight" if first.is_packed else target
        )
        layout[target] = replace(
            first,
            logical_shape=(
                sum(m.logical_shape[0] for m in metas),
                *first.logical_shape[1:],
            ),
            stored_shape=(
                sum(m.stored_shape[0] for m in metas),
                *first.stored_shape[1:],
            ),
            param_name=param_name,
        )
        for index, (name, meta) in parts.items():
            layout[name] = meta
            shard_plan[meta.param_name] = (param_name, index, len(parts))
    return layout, shard_plan


def _runtime_gguf_weights(weights, layout: dict, shard_plan: dict, quant_config):
    """Fuse planned shards; dequantize packed tensors whose layer built without
    GGUF (e.g. Z-Image adaLN ReplicatedLinear). Runs after model construction,
    so quant_config.selected is final."""
    import gguf

    packed_types = {m.param_name: m.weight_type for m in layout.values() if m.is_packed}
    pending = defaultdict(dict)
    for name, tensor in weights:
        if name in shard_plan:
            name, index, total = shard_plan[name]
            pending[name][index] = tensor
            if len(pending[name]) < total:
                continue
            parts = pending.pop(name)
            tensor = torch.cat([parts[i] for i in range(total)])
        prefix = name.removesuffix(".qweight")
        if name in packed_types and prefix not in quant_config.selected:
            value = gguf.dequantize(tensor.numpy(), packed_types[name])
            name, tensor = f"{prefix}.weight", torch.from_numpy(value).bfloat16()
        yield name, tensor


def _load_comfyui_gguf_transformer(
    *,
    model_cls,
    dit_config,
    gguf_file: str,
    server_args: ServerArgs,
    mapping: dict[str, Any],
    spec,
):
    from sglang.multimodal_gen.runtime.layers.quantization.gguf import GGUFConfig
    from sglang.multimodal_gen.runtime.loader.gguf_weights import (
        gguf_weights_iterator,
        read_gguf_tensor_meta,
    )
    from sglang.multimodal_gen.runtime.loader.minimax_h3_weights import (
        validate_minimax_h3_checkpoint_variant,
    )
    from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
        _validate_gguf_runtime_support,
    )

    _validate_gguf_runtime_support(server_args, "transformer")
    if spec.dit_cls_name == "MiniMaxH3DiTModel":
        validate_minimax_h3_checkpoint_variant(
            [gguf_file], str(server_args.model_variant or "fl2va")
        )

    # Quant lookup and weight loading must both see runtime names.
    tensor_meta, shard_plan = _gguf_runtime_layout(
        read_gguf_tensor_meta(gguf_file), mapping
    )
    curve = tensor_meta.get("adaln_t_table")
    if curve is not None:
        if curve.is_quantized or len(curve.logical_shape) != 2:
            raise ValueError(
                "MiniMax-H3 adaln_t_table must be an unquantized 2D tensor"
            )
        curve_grid, time_embed_dim = curve.logical_shape
        if curve_grid < 2:
            raise ValueError("MiniMax-H3 adaln_t_table needs at least two rows")
        dit_config.arch_config.adaln_curve_grid = curve_grid
        dit_config.arch_config.time_embed_dim = time_embed_dim

    quant_config = GGUFConfig(gguf_file=gguf_file, tensor_meta=tensor_meta)
    packed = getattr(model_cls, "packed_modules_mapping", None)
    if packed:
        quant_config.packed_modules_mapping = packed

    logger.info("Loading %s from GGUF file %s", spec.dit_cls_name, gguf_file)
    original_mapping = model_cls.param_names_mapping
    model_cls.param_names_mapping = mapping
    try:
        return maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={
                "config": dit_config,
                "hf_config": {},
                "quant_config": quant_config,
            },
            weight_dir_list=[],
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            component_starts_on_cpu=server_args.should_start_component_on_cpu(
                "transformer"
            ),
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.should_use_fsdp_for_component("transformer"),
            param_dtype=None,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
            weights_iterator=_runtime_gguf_weights(
                gguf_weights_iterator(gguf_file, tensor_meta),
                tensor_meta,
                shard_plan,
                quant_config,
            ),
        )
    finally:
        model_cls.param_names_mapping = original_mapping


def _resolve_online_quant_spec(
    *,
    server_args: ServerArgs,
    model_cls,
    cls_name: str,
    model_path: str,
    dit_config,
):
    from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
        resolve_transformer_quant_load_spec,
    )

    if server_args.quantization not in _ONLINE_QUANTIZATIONS:
        raise ValueError(
            f"--quantization {server_args.quantization} needs a pre-quantized "
            "checkpoint; a BF16 ComfyUI file can only be quantized online with "
            f"one of {sorted(_ONLINE_QUANTIZATIONS)}"
        )
    return resolve_transformer_quant_load_spec(
        hf_config={},
        server_args=server_args,
        safetensors_list=[model_path],
        component_model_path=os.path.dirname(model_path) or ".",
        model_cls=model_cls,
        cls_name=cls_name,
        component_name="transformer",
        arch_config=dit_config.arch_config,
    )


def _online_quant_load_plan(*, server_args: ServerArgs, quant_spec) -> WeightLoadPlan:
    starts_on_cpu = server_args.should_start_component_on_cpu("transformer")
    # Matches the native loader: only ConvRot INT8 quantizes from host weights.
    checkpoint_load_device = (
        torch.device("cpu")
        if starts_on_cpu and quant_spec.is_convrot_int8
        else get_local_torch_device()
    )
    return WeightLoadPlan.for_component(
        checkpoint_load_device=checkpoint_load_device,
        needs_device_weight_postprocess=quant_spec.needs_device_weight_postprocess,
        component_starts_on_cpu=starts_on_cpu,
    )


def load_comfyui_transformer(
    pipeline: ComposedPipelineBase,
    server_args: ServerArgs,
    loaded_modules: dict[str, torch.nn.Module] | None = None,
) -> dict[str, Any]:
    """Load the DiT from a single ComfyUI safetensors file.

    Reuses the shared FSDP-aware transformer load path; the spec only supplies
    what the file itself cannot describe. A GGUF
    ``--transformer-weights-path`` replaces only the DiT weights.
    """
    if loaded_modules is not None and "transformer" in loaded_modules:
        return {
            "transformer": loaded_modules["transformer"],
            "scheduler": pipeline.get_module("scheduler"),
        }

    spec = get_comfyui_checkpoint_spec(pipeline.pipeline_name)
    if spec is None:
        raise ValueError(
            f"{pipeline.pipeline_name} has no ComfyUI checkpoint spec, so it cannot "
            f"load a single safetensors file. Pipelines with a spec: "
            f"{get_registered_comfyui_pipeline_names()}"
        )

    model_path = pipeline.model_path
    dit_config = spec.build_dit_config(server_args)
    mapping = dict(spec.param_names_mapping)
    if spec.inherit_config_mapping:
        mapping = {
            **(dit_config.arch_config.param_names_mapping or {}),
            **mapping,
        }
    dit_config.arch_config.param_names_mapping = mapping

    model_cls, _ = ModelRegistry.resolve_model_cls(spec.dit_cls_name)
    param_dtype = resolve_precision(server_args, "dit", precision_attr="dit_precision")
    server_args.model_paths["transformer"] = os.path.dirname(model_path) or "."

    from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
        resolve_transformer_gguf_to_load,
    )

    gguf_file = resolve_transformer_gguf_to_load(server_args, "transformer")
    if gguf_file is None and model_path.endswith(".gguf"):
        # A GGUF unet picked in ComfyUI is the model path itself.
        gguf_file = model_path
    if gguf_file is not None:
        model = _load_comfyui_gguf_transformer(
            model_cls=model_cls,
            dit_config=dit_config,
            gguf_file=gguf_file,
            server_args=server_args,
            mapping=mapping,
            spec=spec,
        )
    else:
        # Only override the iterator when tensors need reshaping; leaving it None
        # keeps the rank-local checkpoint fast path available.
        weights_iterator = None
        if spec.convert_weights is not None:
            weights_iterator = spec.convert_weights(
                safetensors_weights_iterator([model_path]), dit_config
            )

        quant_config = None
        checkpoint_key_filter = spec.checkpoint_key_filter
        weight_load_plan = None
        # GGUF already sets AdaLN curve from tensor meta. Pruned BF16
        # safetensors keep the same adaln_t_table; without this the DiT is
        # built as the unpruned MLP and load fails on that extra parameter.
        if spec.dit_cls_name == "MiniMaxH3DiTModel" or spec.quant_markers:
            if spec.quant_markers:
                # Format support is checked by the quantization resolver below.
                adaln_curve_shape = None
                layer_markers = spec.quant_markers([model_path])
            else:
                adaln_curve_shape, layer_markers = inspect_minimax_h3_safetensors(
                    [model_path]
                )
            if layer_markers:
                if not spec.quant_markers and any(
                    marker.get("format") != "int8_tensorwise"
                    for marker in layer_markers.values()
                ):
                    raise ValueError(
                        "ComfyUI H3 integrated mode currently supports serialized INT8 ConvRot quantization"
                    )
                if (
                    server_args.quantization is not None
                    or server_args.nunchaku_config is not None
                ):
                    raise ValueError(
                        "Checkpoint quantization is encoded in per-layer metadata; do not also set quantization or Nunchaku"
                    )
                if server_args.should_use_fsdp_for_component("transformer"):
                    raise ValueError(
                        "Comfy quantized checkpoints do not support FSDP inference; use TP and/or sequence parallelism instead"
                    )
                quant_config = resolve_minimax_h3_checkpoint_quantization(layer_markers)
                spec_filter = spec.checkpoint_key_filter
                checkpoint_key_filter = (
                    comfy_quant_key_filter
                    if spec_filter is None
                    else lambda key: comfy_quant_key_filter(key) and spec_filter(key)
                )
                checkpoint_device = (
                    torch.device("cpu")
                    if server_args.should_start_component_on_cpu("transformer")
                    else get_local_torch_device()
                )
                weight_load_plan = WeightLoadPlan(
                    checkpoint_load_device=checkpoint_device
                )
            if adaln_curve_shape is not None:
                (
                    dit_config.arch_config.adaln_curve_grid,
                    dit_config.arch_config.time_embed_dim,
                ) = adaln_curve_shape
                logger.info(
                    "MiniMax-H3 ComfyUI checkpoint uses AdaLN curve %s",
                    adaln_curve_shape,
                )

        post_load_hooks = []
        if quant_config is None and server_args.quantization is not None:
            # A BF16 file has no markers; quantize it online like a native directory.
            quant_spec = _resolve_online_quant_spec(
                server_args=server_args,
                model_cls=model_cls,
                cls_name=spec.dit_cls_name,
                model_path=model_path,
                dit_config=dit_config,
            )
            quant_config = quant_spec.quant_config
            param_dtype = quant_spec.param_dtype
            post_load_hooks = quant_spec.post_load_hooks
            weight_load_plan = _online_quant_load_plan(
                server_args=server_args, quant_spec=quant_spec
            )

        logger.info(
            "Loading %s from ComfyUI checkpoint %s, param_dtype: %s",
            spec.dit_cls_name,
            model_path,
            param_dtype,
        )

        # Weight loading reads param_names_mapping off the model, which inherits it
        # from the class, so the ComfyUI names have to be visible for the whole load.
        init_params = {"config": dit_config, "hf_config": {}}
        if quant_config is not None:
            init_params["quant_config"] = quant_config
        original_mapping = model_cls.param_names_mapping
        model_cls.param_names_mapping = mapping
        try:
            model = maybe_load_fsdp_model(
                model_cls=model_cls,
                init_params=init_params,
                weight_dir_list=[model_path],
                device=get_local_torch_device(),
                hsdp_replicate_dim=server_args.hsdp_replicate_dim,
                hsdp_shard_dim=server_args.hsdp_shard_dim,
                component_starts_on_cpu=server_args.should_start_component_on_cpu(
                    "transformer"
                ),
                pin_cpu_memory=server_args.pin_cpu_memory,
                fsdp_inference=server_args.should_use_fsdp_for_component("transformer"),
                param_dtype=param_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                strict=spec.strict,
                weights_iterator=weights_iterator,
                checkpoint_key_filter=checkpoint_key_filter,
                weight_load_plan=weight_load_plan,
            )
        finally:
            model_cls.param_names_mapping = original_mapping
        for post_load_hook in post_load_hooks:
            post_load_hook(model)

    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        "Loaded transformer with %.2fB parameters",
        sum(p.numel() for p in model.parameters()) / 1e9,
    )

    return {"transformer": model, "scheduler": pipeline.get_module("scheduler")}
