# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py

from __future__ import annotations

import re
from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union

import numpy
import torch

from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


def get_scalar_types():
    """
    Returns:
        tuple: (ScalarType, scalar_types)
    """
    try:
        from sgl_kernel.scalar_type import ScalarType, scalar_types

        return ScalarType, scalar_types
    except ImportError:

        class MockScalarType:
            pass

        class MockScalarTypes:
            uint4b8 = "uint4b8"
            uint8b128 = "uint8b128"

            def __getattr__(self, name):
                return f"mock_{name}"

        return MockScalarType, MockScalarTypes()


ScalarType, scalar_types = get_scalar_types()


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
            is_shard_skipped = any(
                ignored in shard_prefix for ignored in ignored_layers
            )

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = any(ignored in prefix for ignored in ignored_layers)
        if "gate_up_proj" in prefix:
            prefix_gate = prefix.replace("gate_up_proj", "gate_proj")
            prefix_up = prefix.replace("gate_up_proj", "up_proj")
            if prefix_gate in ignored_layers and prefix_up in ignored_layers:
                is_skipped = True
        elif "experts" in prefix:
            is_skipped = any(
                [
                    prefix in layer_name
                    for layer_name in ignored_layers
                    if "experts" in layer_name
                ]
            )

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


def update_tensor_inplace(old: torch.Tensor, new: torch.Tensor) -> None:
    old.copy_(new)


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


def assert_fp8_all_close(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    assert a.dtype == b.dtype == torch.float8_e4m3fn

    a_u8 = a.view(torch.uint8)
    b_u8 = b.view(torch.uint8)
    diff_u8 = (a_u8.to(torch.int16) - b_u8.to(torch.int16)).abs()

    numel = a.numel()

    count_diff_sign = ((a_u8 >= 0) & (b_u8 < 0)).sum().item()
    count_tiny_diff = (diff_u8 >= 1).sum().item()
    count_large_diff = (diff_u8 >= 2).sum().item()

    assert (
        (count_diff_sign == 0)
        and (count_tiny_diff / numel < 0.005)
        and (count_large_diff == 0)
    ), f"{count_diff_sign=} {count_tiny_diff=} {count_large_diff=} {numel=}"


# Match dynamic rules with module name (prefix) and override quantize
# config if module (prefix) matches a rule
def override_config(config: QuantizationConfig, prefix: str):
    weight_bits = get_dynamic_override(config, prefix, "bits", config.weight_bits)
    if isinstance(weight_bits, int):
        config.weight_bits = weight_bits
    group_size = get_dynamic_override(config, prefix, "group_size", config.group_size)
    if isinstance(group_size, int):
        config.group_size = group_size
    desc_act = get_dynamic_override(config, prefix, "desc_act", config.desc_act)
    if isinstance(desc_act, bool):
        config.desc_act = desc_act

    config.pack_factor = 32 // config.weight_bits  # packed into int32
    if config.get_name() == "gptq_marlin":
        is_sym = get_dynamic_override(config, prefix, "sym", config.is_sym)
        if isinstance(is_sym, bool):
            config.is_sym = is_sym

        if (config.weight_bits, config.is_sym) not in config.TYPE_MAP:
            raise ValueError(
                "Unsupported quantization config: "
                f"bits={config.weight_bits}, sym={config.is_sym}"
            )

        config.quant_type = config.TYPE_MAP[(config.weight_bits, config.is_sym)]
    elif config.get_name() == "gptq":
        if config.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {config.weight_bits} bits."
            )


def get_dynamic_override(
    config: QuantizationConfig,
    layer_name: str,
    key: Optional[str] = None,
    default_value: Union[int, bool, None] = None,
) -> Union[Dict, int, bool, None]:
    for pattern, pattern_dict in config.dynamic.items():
        # Negative match: matched modules are excluded from quantized init
        if pattern.startswith("-:"):
            if re.match(pattern.removeprefix("-:"), layer_name):
                return False
        # Positive match: matched modules have quant properties overrides
        # base quant config
        elif re.match(pattern.removeprefix("+:"), layer_name):
            if key is None:
                return pattern_dict
            else:
                return pattern_dict.get(key, default_value)
    return default_value


def get_linear_quant_method(
    config: QuantizationConfig,
    layer: torch.nn.Module,
    prefix: str,
    linear_method_cls: type,
):
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedEmbeddingMethod,
        UnquantizedLinearMethod,
    )
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    cloned_config = deepcopy(config)
    parallel_lm_head_quantized = (
        isinstance(layer, ParallelLMHead) and cloned_config.lm_head_quantized
    )

    if isinstance(layer, LinearBase) or parallel_lm_head_quantized:
        # False = skip module, None = no override, else = Positive match
        if get_dynamic_override(cloned_config, layer_name=prefix) is False:
            if parallel_lm_head_quantized:
                return UnquantizedEmbeddingMethod()
            return UnquantizedLinearMethod()

        if prefix:
            # Dynamic per module/layer rules may override base config
            override_config(cloned_config, prefix=prefix)

        return linear_method_cls(cloned_config)
    return None


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def permute_rows(
    q_w: torch.Tensor,
    w_ref: torch.Tensor,
    group_size: int,
    test_perm: Optional[torch.Tensor] = None,
):
    assert q_w.shape == w_ref.shape

    orig_device = q_w.device
    k_size, _ = q_w.shape

    g_idx = torch.zeros((k_size,), dtype=torch.int32)
    for i in range(k_size):
        g_idx[i] = i // group_size

    # Simulate act_order by doing a random permutation on K
    rand_perm = test_perm if test_perm is not None else torch.randperm(k_size)

    g_idx = g_idx[rand_perm].contiguous()
    q_w = q_w[rand_perm, :].contiguous()
    w_ref = w_ref[rand_perm, :].contiguous()

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )


def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def pack_rows(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k // pack_factor, size_n), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[i::pack_factor, :] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res


def unpack_cols(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (
        size_k,
        size_n // pack_factor,
    ), "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
        packed_q_w.shape, size_k, size_n, pack_factor
    )

    orig_device = packed_q_w.device

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py
def quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: Optional[int],
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    assert (
        quant_type.is_integer()
    ), "Floating point quantization may work but has not been tested"
    assert not zero_points or group_size is not None, (
        "to have group zero points, group_size must be provided "
        "(-1 group_size is channelwise)"
    )

    orig_device = w.device
    orig_type = w.dtype
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    # Reshape to [groupsize, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    max_val = torch.max(w, 0, keepdim=True).values
    min_val = torch.min(w, 0, keepdim=True).values

    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = torch.Tensor([1.0]).to(w.device)  # unscaled case
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            maybe_w_zp = (
                torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()
            )
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),
            )

    # Quantize
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    # Compute ref (dequantized)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
    if ref_zero_points_after_scales and maybe_w_zp is not None:
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )


SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


def gptq_quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: Optional[torch.Tensor] = None,
):
    size_k, _ = w.shape

    assert w.is_floating_point(), "w must be float"
    assert (
        quant_type in SUPPORTED_GPTQ_QUANT_TYPES
    ), f"Unsupported gptq type = {quant_type}"
    assert group_size in SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    w_ref, w_q, w_s, _ = quantize_weights(w, quant_type, group_size)

    # Apply act_order
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        assert (
            group_size < size_k
        ), "For act_order, groupsize = {} must be less than size_k = {}".format(
            group_size, size_k
        )

        w_ref, w_q, g_idx, rand_perm = permute_rows(w_q, w_ref, group_size, test_perm)

    return w_ref, w_q, w_s, g_idx, rand_perm


def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):
    orig_device = q_w.device

    sort_indices = torch.argsort(g_idx).to(dtype=torch.int32)  # Sort based on g_idx

    g_idx = g_idx[sort_indices].contiguous()
    q_w = q_w[sort_indices, :].contiguous()

    return (
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        sort_indices.to(device=orig_device),
    )
