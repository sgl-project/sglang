# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import Cache

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models import (  # noqa: F401
    sensenova_u1 as _sensenova_u1,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3Model,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Arbitrary; leaves room for 2K-resolution CFG activations and the KV prefix.
WEIGHT_MEMORY_FRACTION = 0.75

# Keyword values Qwen3Model.forward passes to each decoder layer.
DecoderLayerInput = torch.Tensor | bool | Cache | None


def load_model_and_tokenizer(
    model_path: str,
    server_args: ServerArgs,
) -> dict[str, Any]:
    dtype = PRECISION_TO_TYPE.get(
        server_args.pipeline_config.model_precision, torch.bfloat16
    )
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if server_args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if server_args.revision is not None:
        model_kwargs["revision"] = server_args.revision

    tokenizer_kwargs: dict[str, Any] = {}
    if server_args.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    if server_args.revision is not None:
        tokenizer_kwargs["revision"] = server_args.revision

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
    device = get_local_torch_device()
    current_platform.set_device(device)
    model = place_model(model=model, device=device)
    return {"model": model, "tokenizer": tokenizer}


def module_weight_bytes(module: torch.nn.Module) -> int:
    tensors = [*module.parameters(), *module.buffers()]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def place_model(model: NEOChatModel, device: torch.device) -> NEOChatModel:
    weight_budget_bytes = int(
        current_platform.get_device_total_memory(device.index or 0)
        * WEIGHT_MEMORY_FRACTION
    )
    total_weight_bytes = module_weight_bytes(model)
    required_device_count = math.ceil(total_weight_bytes / weight_budget_bytes)
    first_device_index = device.index or 0
    visible_device_count = (
        torch.get_device_module(device.type).device_count() - first_device_index
    )
    if required_device_count <= 1:
        return model.to(device)
    elif required_device_count > visible_device_count:
        raise ValueError(
            f"SenseNova-U1 weights need {total_weight_bytes / 2**30:.1f} GiB across "
            f"{required_device_count} devices, but only {visible_device_count} are "
            f"visible from {device}; expose more devices or use larger ones."
        )
    else:
        # The pipeline has no tensor parallelism, so oversized weights are split by layer.
        split_devices = [
            torch.device(device.type, first_device_index + offset)
            for offset in range(required_device_count)
        ]
        decoder = model.language_model.model
        layers = decoder.layers
        layer_devices = assign_layer_devices(
            layer_weight_bytes=[module_weight_bytes(layer) for layer in layers],
            primary_weight_bytes=total_weight_bytes
            - sum(module_weight_bytes(layer) for layer in layers),
            devices=split_devices,
        )
        # Detach the layers so the primary device never holds all of them.
        decoder.layers = torch.nn.ModuleList()
        model.to(device)
        decoder.layers = layers
        for layer, layer_device in zip(layers, layer_devices):
            layer.to(layer_device)
        _install_device_hooks(decoder=decoder, layer_devices=layer_devices)
        layer_counts = [layer_devices.count(split) for split in split_devices]
        logger.info(
            f"SenseNova-U1 weights ({total_weight_bytes / 2**30:.1f} GiB) exceed one "
            f"device; decoder layers split over {[str(split) for split in split_devices]} "
            f"as {layer_counts}"
        )
        return model


def assign_layer_devices(
    layer_weight_bytes: list[int],
    primary_weight_bytes: int,
    devices: list[torch.device],
) -> list[torch.device]:
    # Non-layer weights stay on the primary device; later layers fill devices in order.
    bytes_per_device = (primary_weight_bytes + sum(layer_weight_bytes)) / len(devices)
    layer_devices: list[torch.device] = []
    device_index, assigned_bytes = 0, primary_weight_bytes
    for layer_bytes in layer_weight_bytes:
        if (
            assigned_bytes + layer_bytes > bytes_per_device
            and device_index < len(devices) - 1
        ):
            device_index, assigned_bytes = device_index + 1, 0
        else:
            pass
        layer_devices.append(devices[device_index])
        assigned_bytes += layer_bytes
    return layer_devices


def _install_device_hooks(
    decoder: Qwen3Model, layer_devices: list[torch.device]
) -> None:
    primary_device = layer_devices[0]
    device_by_layer = {
        layer: layer_device
        for layer, layer_device in zip(decoder.layers, layer_devices)
        if layer_device != primary_device
    }
    segment_start_layers = {
        layer
        for layer_index, (layer, layer_device) in enumerate(
            zip(decoder.layers, layer_devices)
        )
        if layer in device_by_layer and layer_devices[layer_index - 1] != layer_device
    }
    # Every layer gets the same mask and index tensors; copy each once per device segment.
    copied_inputs_by_device: dict[
        torch.device, dict[int, tuple[torch.Tensor, torch.Tensor]]
    ] = {}

    def move_layer_inputs(
        layer: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, DecoderLayerInput],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, DecoderLayerInput]]:
        layer_device = device_by_layer[layer]
        if layer in segment_start_layers:
            copied_inputs_by_device.pop(layer_device, None)
        else:
            pass
        copied_inputs = copied_inputs_by_device.setdefault(layer_device, {})
        moved_kwargs: dict[str, DecoderLayerInput] = {}
        for name, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                moved_kwargs[name] = value
            elif id(value) in copied_inputs and copied_inputs[id(value)][0] is value:
                moved_kwargs[name] = copied_inputs[id(value)][1]
            else:
                moved_kwargs[name] = value.to(layer_device, non_blocking=True)
                copied_inputs[id(value)] = (value, moved_kwargs[name])
        hidden_states = args[0].to(layer_device, non_blocking=True)
        return (hidden_states, *args[1:]), moved_kwargs

    def move_norm_input_to_primary(
        norm: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        # norm is required by the forward pre-hook signature.
        return tuple(hidden_states.to(primary_device) for hidden_states in args)

    for layer in device_by_layer:
        layer.register_forward_pre_hook(move_layer_inputs, with_kwargs=True)
    for norm in (decoder.norm, decoder.norm_mot_gen):
        norm.register_forward_pre_hook(move_norm_input_to_primary)
