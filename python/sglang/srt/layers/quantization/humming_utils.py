from __future__ import annotations

from typing import TYPE_CHECKING, Any

import regex as re
import torch
from humming.layer import HummingInputSchema, HummingMethod
from humming.schema import BaseWeightSchema, HummingWeightSchema

from sglang.srt.environ import envs
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


def humming_is_layer_skipped(config: dict[str, Any], prefix: str):
    if not config:
        return True

    keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
    ignored_layers: list[str] = []
    for key in keys:
        if key in config:
            ignored_layers = config[key] or []
            break

    if any(module_name in prefix for module_name in ignored_layers):
        return True
    if "lm_head" in prefix:
        return True

    for regex in config.get("dynamic", {}):
        if regex[:1] != "-":
            continue
        if re.match(regex[2:], prefix):
            return True

    return False


def _set_humming_dispatcher_output_dtype(
    layer: torch.nn.Module, output_dtype: str
) -> None:
    dispatcher = getattr(layer, "dispatcher", None)
    if dispatcher is None:
        return

    quant_config = dict(getattr(dispatcher, "quant_config", None) or {})
    quant_config["dispatcher_output_dtype"] = output_dtype
    dispatcher.set_quant_config(quant_config)


def configure_humming_deepep_dispatch(layer: torch.nn.Module) -> bool:
    if not get_moe_a2a_backend().is_deepep():
        layer._humming_uses_deepep_fp8_dispatch = False
        _set_humming_dispatcher_output_dtype(layer, "bf16")
        return False

    output_dtype = get_exec().moe.deepep_dispatcher_output_dtype
    if output_dtype == "auto":
        output_dtype = "bf16" if envs.SGLANG_DEEPEP_BF16_DISPATCH.get() else "fp8"
    if output_dtype not in ("bf16", "fp8"):
        raise ValueError(
            f"Humming does not support DeepEP {output_dtype} dispatch; "
            "use --deepep-dispatcher-output-dtype=bf16 or fp8."
        )

    _set_humming_dispatcher_output_dtype(layer, output_dtype)
    use_fp8 = output_dtype == "fp8"
    layer._humming_uses_deepep_fp8_dispatch = use_fp8
    return use_fp8


def make_humming_deepep_input_schema(
    sublayer_name: str,
    shape_k: int,
    *,
    weight_schema: HummingWeightSchema | None = None,
    param_dtype: torch.dtype | None = None,
) -> HummingInputSchema:
    if shape_k % 128 != 0:
        raise ValueError(
            f"Humming FP8 dispatch requires {sublayer_name} K={shape_k} "
            "to be divisible by 128."
        )
    input_schema = HummingInputSchema(
        a_dtype="float8e4m3",
        input_scale_group_size=128,
        input_scale_dtype="float32",
        input_quant_mode="dynamic_group",
    )
    if weight_schema is not None and not input_schema.is_compatible_with(
        weight_schema, param_dtype
    ):
        raise ValueError(
            f"Humming FP8 dispatch is incompatible with {sublayer_name} weights "
            "on this device: it requires FP8 inputs with FP32 group-128 scales. "
            "Use --deepep-dispatcher-output-dtype=bf16."
        )
    return input_schema


def prepare_humming_moe_layer(layer: FusedMoE, quant_config: dict):
    weight_schema = BaseWeightSchema.from_config(quant_config)
    input_quant_config = envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG.get() or {}
    if humming_is_layer_skipped(input_quant_config, layer.layer_name):
        input_schema = HummingInputSchema()
    else:
        # TODO: read input_quant_config from quant_config
        input_schema = HummingInputSchema.from_config(input_quant_config)

    use_deepep_fp8_dispatch = configure_humming_deepep_dispatch(layer)

    shape_config = {
        "w13": (
            layer.intermediate_size_per_partition * 2,
            layer.hidden_size,
        ),
        "w2": (
            layer.hidden_size,
            layer.intermediate_size_per_partition,
        ),
    }

    layer.weight_schemas = {}
    layer.input_schemas = {}

    for sublayer_name, (shape_n, shape_k) in shape_config.items():
        _prepare_humming_moe_sublayer(
            layer=layer,
            sublayer_name=sublayer_name,
            shape_n=shape_n,
            shape_k=shape_k,
            weight_schema=weight_schema,
            input_schema=input_schema,
            use_deepep_fp8_dispatch=use_deepep_fp8_dispatch,
        )

    if not hasattr(layer, "locks"):
        device = layer.w13_weight.device
        locks = torch.zeros(1024, dtype=torch.int32, device=device)
        layer.register_buffer("locks", locks)


def _prepare_humming_moe_sublayer(
    layer: FusedMoE,
    sublayer_name: str,
    shape_n: int,
    shape_k: int,
    weight_schema: BaseWeightSchema,
    input_schema: HummingInputSchema,
    use_deepep_fp8_dispatch: bool,
) -> None:
    # Release old-weight references before the next sublayer.
    # Step 1: convert weight to humming standard format
    checkpoint_tensors: dict[str, torch.Tensor] = dict(
        (key.removeprefix(sublayer_name + "_"), value)
        for key, value in layer.state_dict().items()
        if key.startswith(sublayer_name + "_")
    )

    shape_n_stacks = [shape_n]
    shape_k_stacks = [shape_k]
    if sublayer_name == "w13":
        shape_n_stacks = [shape_n // 2] * 2

    weight_schema_new, tensors = weight_schema.convert_humming(
        tensors=checkpoint_tensors,
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        num_experts=layer.num_local_experts,
        param_dtype=layer.params_dtype,
    )
    tensors = tensors.copy()
    tensors.pop("input_scale", None)
    tensors.pop("input_scale_2", None)

    layer.weight_schemas[sublayer_name] = weight_schema_new
    if use_deepep_fp8_dispatch:
        sub_input_schema = make_humming_deepep_input_schema(
            sublayer_name,
            shape_k,
            weight_schema=weight_schema_new,
            param_dtype=layer.params_dtype,
        )
    else:
        sub_input_schema, input_tensors = input_schema.convert_humming(
            tensors=checkpoint_tensors,
            shape_n_stacks=shape_n_stacks,
            shape_k_stacks=shape_k_stacks,
            num_experts=layer.num_local_experts,
            param_dtype=layer.params_dtype,
        )
        tensors.update(input_tensors)
    layer.input_schemas[sublayer_name] = sub_input_schema

    # Preparing integer activations can replace E0Mx weight tensors. Register
    # the resulting tensors only after this conversion has completed.
    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=shape_n,
        shape_k=shape_k,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        input_schema=sub_input_schema,
        weight_schema=weight_schema_new,
        has_bias=layer.with_bias,
        num_experts=layer.num_local_experts,
        torch_dtype=layer.params_dtype,
        sublayer_name=sublayer_name,
        tensors=tensors,
    )

    for name, _ in list(layer.named_parameters()):
        if not name.startswith(sublayer_name + "_"):
            continue
        delattr(layer, name)

    for name, tensor in tensors.items():
        name = f"{sublayer_name}_{name}"
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, name, param)

    # Step 2: transform weight (humming standard format) for forwarding
    HummingMethod.transform_humming_layer(layer, sublayer_name=sublayer_name)
