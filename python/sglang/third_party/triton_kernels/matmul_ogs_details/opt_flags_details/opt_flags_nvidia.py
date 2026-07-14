import torch
import triton
from sglang.third_party.triton_kernels import target_info
from sglang.third_party.triton_kernels.numerics_details.mxfp_details._downcast_to_mxfp import MXFP_BLOCK_SIZE
from sglang.third_party.triton_kernels.tensor import FP4, bitwidth, get_layout
from sglang.third_party.triton_kernels.tensor_details.layout import HopperMXScaleLayout


def compute_grid_size(routing_data, batch_size, m, n, block_m, block_n):
    if routing_data is not None and batch_size == 1:
        grid_m = routing_data.n_blocks(m, block_m)
    else:
        grid_m = triton.cdiv(m, block_m)
    grid_n = (n + block_n - 1) // block_n
    return batch_size * grid_m * grid_n


def compute_block_n(n: int, arch, precision_config):
    # block_n:
    layout = get_layout(precision_config.weight_scale)
    if isinstance(layout, HopperMXScaleLayout):
        if layout.num_warps in [4, 8]:
            # https://github.com/triton-lang/triton/blob/814b862166c756d9f33238844f4ac047e0243388/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py#L265
            block_n = 2 * layout.num_warps * 2 * 8
            return block_n, block_n
    elif precision_config.max_num_imprecise_acc is None and n > 128:
        return 256, 256
    else:
        target = min(128, triton.next_power_of_2(n))
        return max(8, target), max(16, target)


def compute_block_k(m: int, k: int | None, is_persistent: bool, lhs_dtype, rhs_dtype, precision_config, has_y_acc_in):
    lhs_width = bitwidth(lhs_dtype)
    rhs_width = bitwidth(rhs_dtype)
    # block_k needs to match the cacheline size (1024 bits)
    block_k = int(1024 // min(lhs_width, rhs_width))
    has_native_mxfp = target_info.cuda_capability_geq(10, 0)
    if rhs_width == 4 and not has_native_mxfp:
        block_k = 128
    elif k is not None:
        min_block_k = 32 if is_persistent or lhs_width != 16 or rhs_width != 16 else 16
        block_k = max(min_block_k, min(triton.next_power_of_2(k), block_k))
    has_mx_weight_scale = precision_config is not None and precision_config.weight_scale is not None
    if has_native_mxfp and is_persistent and has_mx_weight_scale:
        block_k = min(block_k, 128)
    if has_y_acc_in and lhs_width == rhs_width == 16 and not target_info.cuda_capability_geq(10, 0):
        block_k = min(block_k, 32)
    return block_k


def compute_split_k(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // grid_size
    if k is not None:
        # avoid split_k for small k
        num_block_k = triton.cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    split_k = max(split_k, 1)
    return split_k


def compute_num_warps(block_m, block_n, is_persistent: bool, precision_config):
    layout = get_layout(precision_config.weight_scale)
    if isinstance(layout, HopperMXScaleLayout):
        return layout.num_warps
    return max(block_m * block_n // 4096, 4 if is_persistent else 1)


def compute_num_stages(
    precision_config,
    is_persistent,
    block_m,
    block_n,
    block_k,
    out_dtype,
    lhs_dtype,
    rhs_dtype,
    x_transpose,
    epilogue_effective_itemsize,
    has_y_acc_in,
    *,
    epilogue_subtile,
):
    if precision_config.max_num_imprecise_acc is not None:
        return 3
    weight_size = bitwidth(rhs_dtype) / 8
    stage_size = block_m * block_k * lhs_dtype.itemsize + block_k * block_n * weight_size
    device_props = torch.cuda.get_device_properties(0)
    smem_capacity = device_props.shared_memory_per_block_optin
    has_native_mxfp = target_info.cuda_capability_geq(10, 0)
    if has_native_mxfp and getattr(precision_config, "weight_scale", None) is not None:
        if rhs_dtype == FP4:
            # 4-bit e2m1 weights are padded 2x
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#packing-format-used-for-matrix-a-and-b-by-kind-mxf8f6f4-in-shared-memory
            stage_size += block_k * block_n * weight_size

    if is_persistent:
        # Per-stage wait barrier
        stage_size += 8
        out_itemsize = out_dtype.itemsize * (1.25 if has_y_acc_in else 1.0)
        if target_info.cuda_capability_geq(10, 0):
            acc_size = epilogue_effective_itemsize or out_itemsize
        else:
            acc_size = out_itemsize
        if target_info.cuda_capability_geq(10, 0) and epilogue_subtile is not None:
            acc_block_n = block_n // epilogue_subtile
        else:
            acc_block_n = block_n
        # pipelined TMA store local to global, or
        # pipelined layout conversion before store of the accumulator
        # note: layout conversion has some padding
        smem_capacity -= int((block_m + 4) * acc_block_n * acc_size)
        if x_transpose:
            smem_capacity -= block_m * block_k * lhs_dtype.itemsize
        if precision_config.weight_scale is not None:
            # mx scales
            stage_size += block_n * (block_k // int(MXFP_BLOCK_SIZE))
    elif has_native_mxfp:
        # mx scales
        stage_size += block_n * (block_k // int(MXFP_BLOCK_SIZE))
    num_stages = min(4, smem_capacity // int(stage_size))
    return num_stages
