# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/quant_utils.py

from types import MappingProxyType
from typing import List, Mapping, Tuple, Union, Optional, NamedTuple

import numpy
import torch

from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_npu
from sglang.srt.layers.quantization.quant_utils import (
    pack_cols,
    get_pack_factor,
)
from sgl_kernel import (
    ScalarType,
    scalar_types,
)

_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

if not (_is_cuda or _is_npu or (_is_cpu and _is_cpu_amx_available)):
    from vllm._custom_ops import scaled_fp8_quant

class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer ``<major><minor>``.

        It is assumed that the minor version is always a single digit.
        """
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor

def verify_marlin_supported(quant_type: ScalarType,
                            group_size: int,
                            has_zp: bool = False) -> None:
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)
    if not cond:
        assert err_msg is not None
        raise ValueError(err_msg)

# For binary size and compile time, we don't support the same types for with and
#  without runtime zero-point. We support common cases, i.e. AWQ and GPTQ.
#  TODO: we may want to move this into the C++ so its closer to the actual impl
def query_marlin_supported_quant_types(
    has_zp: Optional[bool] = None,
    include_fp_type: bool = True,
    device_capability: Optional[int] = None,
):
    if device_capability is None:
        capability_tuple = torch.cuda.get_device_capability()
        if capability_tuple is None:
            device_capability = -1
        else:
            device_capability = capability_tuple[0] * 10 + capability_tuple[1]
    
    if device_capability < 80:
        return []

    # - has_zp is True: return quant_types that has zero points
    # - has_zp is False: return quant_types that has not zero points
    # - has_zp is None: both
    if has_zp is None:
        types0 = query_marlin_supported_quant_types(False, include_fp_type,
                                                    device_capability)
        types1 = query_marlin_supported_quant_types(True, include_fp_type,
                                                    device_capability)
        return types0 + types1

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4]
    else:
        # GPTQ style, unsigned + symmetric bias
        res = [scalar_types.uint4b8, scalar_types.uint8b128]
        if include_fp_type:
            res += [scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f]
        return res

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


# Marlin utils
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
USE_FP32_REDUCE_DEFAULT = True

def check_marlin_supported(quant_type: ScalarType,
                           group_size: int,
                           has_zp: bool = False,
                           device_capability: Optional[int] = None) -> bool:
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp,
                                      device_capability)
    return cond

def update_tensor_inplace(dst: torch.Tensor, src: torch.Tensor):
    assert dst.dtype == src.dtype, "Tensors must have the same dtype"

    # update tensor shape and stride
    dst.as_strided_(src.shape, src.stride())

    # If not the same underlying storage move tensor data
    if dst.data_ptr() != src.data_ptr():
        dst.copy_(src)
        del src

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

def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]]
        )
    return scale_perm, scale_perm_single

def _check_marlin_supported(
        quant_type: ScalarType,
        group_size: Optional[int],
        has_zp: bool,
        device_capability: Optional[int] = None) -> tuple[bool, Optional[str]]:

    if device_capability is None:
        capability_tuple = torch.cuda.get_device_capability()
        if capability_tuple is None:
            device_capability = -1
        else:
            device_capability = capability_tuple[0] * 10 + capability_tuple[1]

    supported_types = query_marlin_supported_quant_types(
        has_zp, True, device_capability)

    if quant_type not in supported_types:
        return (False, f"Marlin does not support weight_bits = {quant_type}. "
                f"Only types = {supported_types} "
                f"are supported (for group_size = {group_size}, "
                f"device_capability = {device_capability}, zp = {has_zp}).")
    if (group_size is None or group_size not in MARLIN_SUPPORTED_GROUP_SIZES):
        return (False, f"Marlin does not support group_size = {group_size}. "
                f"Only group_sizes = {MARLIN_SUPPORTED_GROUP_SIZES} "
                "are supported.")

    return True, None

def marlin_moe_permute_scales(
    s: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    num_experts = s.shape[0]
    output = torch.empty(
        (num_experts, s.shape[1], s.shape[2]),
        device=s.device,
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output

def should_use_atomic_add_reduce(m: int, n: int, k: int, device: torch.device,
                                 dtype: torch.dtype) -> bool:

    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        return False

    return True

def apply_gptq_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        wtype: ScalarType,
        output_size_per_partition: int,
        input_size_per_partition: int,
        is_k_full: bool,
        bias: Optional[torch.Tensor] = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition, )

    use_atomic_add = should_use_atomic_add_reduce(m=reshaped_x.size(0),
                                                  n=output_size_per_partition,
                                                  k=reshaped_x.size(1),
                                                  device=input.device,
                                                  dtype=input.dtype)

    output = ops.gptq_marlin_gemm(reshaped_x,
                                  None,
                                  weight,
                                  weight_scale,
                                  None,
                                  weight_zp,
                                  g_idx,
                                  g_idx_sort_indices,
                                  workspace,
                                  wtype,
                                  size_m=reshaped_x.shape[0],
                                  size_n=output_size_per_partition,
                                  size_k=input_size_per_partition,
                                  is_k_full=is_k_full,
                                  use_atomic_add=use_atomic_add,
                                  use_fp32_reduce=use_fp32_reduce,
                                  is_zp_float=False)

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)

def verify_marlin_supports_shape(output_size_per_partition: int,
                                 input_size_per_partition: int,
                                 input_size: int, group_size: int) -> None:

    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(f"Weight output_size_per_partition = "
                         f"{output_size_per_partition} is not divisible by "
                         f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(f"Weight input_size_per_partition = "
                         f"{input_size_per_partition} is not divisible "
                         f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
                         "Consider reducing tensor_parallel_size or running "
                         "with --quantization gptq.")

    if (group_size < input_size
            and input_size_per_partition % group_size != 0):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq.")

def check_marlin_supports_shape(output_size_per_partition: int,
                                input_size_per_partition: int,
                                input_size: int, group_size: int) \
                                    -> tuple[bool, Optional[str]]:
    try:
        verify_marlin_supports_shape(output_size_per_partition,
                                     input_size_per_partition, input_size,
                                     group_size)
    except ValueError as e:
        return False, e.__str__()
    return True, None

def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)

def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)

def marlin_make_workspace_new(device: torch.device,
                              max_blocks_per_sm: int = 1) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm,
                       dtype=torch.int,
                       device=device,
                       requires_grad=False)

def marlin_repeat_scales_on_all_ranks(act_order: bool, group_size: int,
                                      is_row_parallel: bool) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)

def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s

def marlin_sort_g_idx(
        g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices

def marlin_zero_points(zp: torch.Tensor, size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp