import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    SpatialParallelConv2d,
    chunk_height_by_sizes,
    gather_and_trim_height,
    gather_variable_height,
    split_for_parallel_decode,
)


def count_decoder_spatial_upsamples(decoder: nn.Module) -> int:
    return sum(
        len(upsamplers)
        for block in getattr(decoder, "up_blocks", [])
        if (upsamplers := getattr(block, "upsamplers", None)) is not None
    )


def enable_diffusers_decoder_spatial_parallel(decoder: nn.Module) -> int:
    _replace_conv2d_modules(decoder)
    _patch_groupnorm_modules(decoder)
    _patch_attention_modules(decoder)
    return count_decoder_spatial_upsamples(decoder)


def spatial_parallel_diffusers_decode(
    decoder: nn.Module, z: torch.Tensor, upsample_count: int
) -> torch.Tensor:
    z, expected_height = split_for_parallel_decode(
        z,
        upsample_count=upsample_count,
        world_size=get_decode_parallel_world_size(),
        rank=get_decode_parallel_rank(),
    )
    return gather_and_trim_height(decoder(z), expected_height)


def _replace_conv2d_modules(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if type(child) is nn.Conv2d:
            setattr(module, name, _make_spatial_conv2d(child))
        else:
            _replace_conv2d_modules(child)


def _make_spatial_conv2d(conv: nn.Conv2d) -> SpatialParallelConv2d:
    spatial_conv = SpatialParallelConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    spatial_conv.weight = conv.weight
    spatial_conv.bias = conv.bias
    return spatial_conv


def _patch_attention_modules(module: nn.Module) -> None:
    for child in module.children():
        if child.__class__.__name__ == "Attention":
            _patch_attention_forward(child)
        _patch_attention_modules(child)


def _patch_groupnorm_modules(module: nn.Module) -> None:
    for child in module.children():
        if type(child) is nn.GroupNorm:
            _patch_groupnorm_forward(child)
        else:
            _patch_groupnorm_modules(child)


def _patch_groupnorm_forward(norm: nn.GroupNorm) -> None:
    original_forward = norm.forward

    def spatial_parallel_forward(hidden_states):
        if hidden_states.dim() < 4:
            return original_forward(hidden_states)
        hidden_states, heights = gather_variable_height(hidden_states)
        hidden_states = hidden_states.contiguous()
        hidden_states = original_forward(hidden_states)
        return chunk_height_by_sizes(hidden_states, heights)

    norm.forward = spatial_parallel_forward


def _patch_attention_forward(attn: nn.Module) -> None:
    original_forward = attn.forward

    def spatial_parallel_forward(hidden_states, *args, **kwargs):
        if hidden_states.dim() != 4:
            return original_forward(hidden_states, *args, **kwargs)
        hidden_states, heights = gather_variable_height(hidden_states)
        hidden_states = hidden_states.contiguous()
        hidden_states = original_forward(hidden_states, *args, **kwargs)
        return chunk_height_by_sizes(hidden_states, heights)

    attn.forward = spatial_parallel_forward
