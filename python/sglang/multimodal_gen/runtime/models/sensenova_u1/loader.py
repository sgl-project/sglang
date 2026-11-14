# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import Cache

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_tp_rank,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.models import (  # noqa: F401
    sensenova_u1 as _sensenova_u1,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
    move_to_device,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3Model,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Per-device weight share of a layer split; arbitrary, leaves room for 2K-resolution
# CFG activations and the KV prefix.
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
    tp_world_size = get_tp_world_size()
    if tp_world_size > 1:
        shard_decoder_for_tensor_parallel(
            model=model, rank=get_tp_rank(), world_size=tp_world_size
        )
    model = place_model(
        model=model, device=device, allow_layer_split=tp_world_size == 1
    )
    return {"model": model, "tokenizer": tokenizer}


def module_weight_bytes(module: torch.nn.Module) -> int:
    tensors = [*module.parameters(), *module.buffers()]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def place_model(
    model: NEOChatModel, device: torch.device, allow_layer_split: bool
) -> NEOChatModel:
    first_device_index = device.index or 0
    device_memory_bytes = current_platform.get_device_total_memory(first_device_index)
    total_weight_bytes = module_weight_bytes(model)
    # Weights that fit one device load there as before; only larger ones are split.
    if total_weight_bytes <= device_memory_bytes:
        return model.to(device)
    required_device_count = math.ceil(
        total_weight_bytes / (device_memory_bytes * WEIGHT_MEMORY_FRACTION)
    )
    visible_device_count = (
        torch.get_device_module(device.type).device_count() - first_device_index
    )
    if not allow_layer_split:
        # Every tensor-parallel rank owns one device, so a shard cannot spill onto the next.
        raise ValueError(
            f"SenseNova-U1 tensor-parallel shard needs {total_weight_bytes / 2**30:.1f} GiB, "
            f"more than {device} holds; raise --tp-size."
        )
    elif required_device_count > visible_device_count:
        raise ValueError(
            f"SenseNova-U1 weights need {total_weight_bytes / 2**30:.1f} GiB across "
            f"{required_device_count} devices, but only {visible_device_count} are "
            f"visible from {device}; expose more devices or use larger ones."
        )
    else:
        # Without tensor parallelism, oversized weights are split by layer.
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
        # Ending each forward on the last device, instead of copying back to the
        # primary, lets the primary start the next CFG branch while the last runs.
        tail_modules = [decoder.norm, decoder.norm_mot_gen, model.fm_modules["fm_head"]]
        for tail_module in tail_modules:
            tail_module.to(layer_devices[-1])
        _install_device_hooks(
            decoder=decoder,
            layer_devices=layer_devices,
            input_moving_modules=[*tail_modules, model.language_model.lm_head],
        )
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
        layer_devices.append(devices[device_index])
        assigned_bytes += layer_bytes
    return layer_devices


def _install_device_hooks(
    decoder: Qwen3Model,
    layer_devices: list[torch.device],
    input_moving_modules: list[torch.nn.Module],
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
        copied_inputs = copied_inputs_by_device.setdefault(layer_device, {})
        moved_kwargs: dict[str, DecoderLayerInput] = {}
        for name, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                moved_kwargs[name] = value
            elif id(value) in copied_inputs and copied_inputs[id(value)][0] is value:
                moved_kwargs[name] = copied_inputs[id(value)][1]
            else:
                moved_kwargs[name] = move_to_device(value, layer_device)
                copied_inputs[id(value)] = (value, moved_kwargs[name])
        hidden_states = move_to_device(args[0], layer_device)
        return (hidden_states, *args[1:]), moved_kwargs

    def move_inputs_to_module_device(
        module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        module_device = next(module.parameters()).device
        return tuple(move_to_device(tensor, module_device) for tensor in args)

    for layer in device_by_layer:
        layer.register_forward_pre_hook(move_layer_inputs, with_kwargs=True)
    # Release the copies once the decoder forward ends, so no request keeps them alive.
    decoder.register_forward_hook(lambda *_: copied_inputs_by_device.clear())
    for module in input_moving_modules:
        module.register_forward_pre_hook(move_inputs_to_module_device)


def shard_decoder_for_tensor_parallel(
    model: NEOChatModel, rank: int, world_size: int
) -> None:
    # Megatron layout: q/k/v and gate/up split by output, o/down by input and summed.
    config = model.language_model.config
    if (
        config.num_attention_heads % world_size
        or config.num_key_value_heads % world_size
        or config.intermediate_size % world_size
    ):
        raise ValueError(
            f"SenseNova-U1 heads ({config.num_attention_heads} query, "
            f"{config.num_key_value_heads} key/value) and intermediate size "
            f"({config.intermediate_size}) must divide by --tp-size {world_size}."
        )
    for layer in model.language_model.model.layers:
        attention = layer.self_attn
        branches = [
            (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.o_proj,
                layer.mlp,
            ),
            (
                attention.q_proj_mot_gen,
                attention.k_proj_mot_gen,
                attention.v_proj_mot_gen,
                attention.o_proj_mot_gen,
                layer.mlp_mot_gen,
            ),
        ]
        for q_proj, k_proj, v_proj, o_proj, mlp in branches:
            if not isinstance(mlp, Qwen3MLP):
                raise ValueError("SenseNova-U1 tensor parallelism needs dense MLPs.")
            for linear in (q_proj, k_proj, v_proj, mlp.gate_proj, mlp.up_proj):
                _keep_shard(linear=linear, rank=rank, world_size=world_size, dim=0)
            for linear in (o_proj, mlp.down_proj):
                _keep_shard(linear=linear, rank=rank, world_size=world_size, dim=1)
                linear.register_forward_hook(_all_reduce_output)
            # The NPU fused MLP splits its packed gate/up weight at this width.
            mlp.intermediate_size = mlp.gate_proj.out_features
    logger.info(
        f"SenseNova-U1 decoder sharded for tensor parallelism: rank {rank} of {world_size}"
    )


def _keep_shard(linear: torch.nn.Linear, rank: int, world_size: int, dim: int) -> None:
    if linear.bias is not None:
        raise ValueError(
            "SenseNova-U1 tensor parallelism expects bias-free projections."
        )
    shard = linear.weight.data.chunk(world_size, dim=dim)[rank].clone()
    linear.weight = torch.nn.Parameter(shard, requires_grad=False)
    linear.out_features, linear.in_features = shard.shape


def _all_reduce_output(
    module: torch.nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> torch.Tensor:
    return tensor_model_parallel_all_reduce(output)
