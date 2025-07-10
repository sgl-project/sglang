# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py

from types import MappingProxyType
from typing import List, Mapping, Tuple, Union

import torch

from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_npu

_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

if not (_is_cuda or _is_npu or (_is_cpu and _is_cpu_amx_available)):
    from vllm._custom_ops import scaled_fp8_quant


def is_layer_skipped(
    prefix: str,
    ignored_layers: List[str],
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
) -> bool:
    # prefix: model.layers.0.self_attn.q_proj
    # proj_name: q_proj
    proj_name = prefix.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = shard_prefix in ignored_layers

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = prefix in ignored_layers

    assert is_skipped is not None
    return is_skipped


def per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def convert_to_channelwise(
    weight_scale: torch.Tensor, logical_widths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create channelwise buffer
    weight_scale_channel = torch.empty(
        (sum(logical_widths), 1), dtype=torch.float32, device=weight_scale.device
    )

    # Handle scalar tensor case: broadcast same scale to all channels
    if weight_scale.dim() == 0:
        weight_scale_channel.fill_(weight_scale.item())
        return weight_scale_channel

    # Expand each scale to match the size of each logical matrix.
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_scale_channel[start:end, :] = weight_scale[idx]
        start = end

    return weight_scale_channel


def requantize_with_max_scale(
    weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    unfused_module_in_checkpoint = (
        weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight[start:end, :], _ = scaled_fp8_quant(weight_dq, max_w_scale)
            start = end

    return max_w_scale, weight


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/layer_utils.py
# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_parameter(
    mod: torch.nn.Module, name: str, new: Union[torch.Tensor, torch.nn.Parameter]
) -> None:

    old = getattr(mod, name)
    if (
        type(old) is type(new)
        and old.dtype == new.dtype
        and old.untyped_storage().nbytes() == new.untyped_storage().nbytes()
    ):
        # If we can just update in-place to avoid re-registering
        #   can be faster if the underlying storage is the same
        update_tensor_inplace(old, new)
    else:
        # Fallback re-register parameter, convert to Parameter if necessary
        # this not only ensures we don't register a tensor as a parameter, but
        # also ensures that all parameter subclasses get re-registered as
        # parameters for `torch.compile` compatibility
        if not isinstance(new, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=False)
        mod.register_parameter(name, torch.nn.Parameter(new, requires_grad=False))
