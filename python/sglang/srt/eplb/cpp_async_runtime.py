from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_ABS_PATH = Path(__file__).resolve().parent
_JIT_CSRC_PATH = _ABS_PATH.parents[1] / "kernels" / "jit" / "csrc" / "eplb"
_JIT_INCLUDE_PATH = _ABS_PATH.parents[1] / "kernels" / "jit" / "include"
_RUNTIME_SOURCES = [
    (_JIT_CSRC_PATH / "eplb_async_runtime.cpp").resolve(),
    (_JIT_CSRC_PATH / "eplb_async_signal_runtime.cu").resolve(),
]

_eplb_async_runtime_cpp = None


def _load_eplb_async_runtime_cpp():
    global _eplb_async_runtime_cpp
    if _eplb_async_runtime_cpp is not None:
        return _eplb_async_runtime_cpp
    missing_sources = [str(path) for path in _RUNTIME_SOURCES if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Missing EPLB async runtime sources: " + ", ".join(missing_sources)
        )
    _eplb_async_runtime_cpp = load(
        name="eplb_async_runtime_cpp",
        sources=[str(path) for path in _RUNTIME_SOURCES],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcudart"],
        extra_include_paths=[str(_JIT_INCLUDE_PATH)],
        with_cuda=True,
    )
    return _eplb_async_runtime_cpp


def create_eplb_async_runtime(device_index: int):
    return _load_eplb_async_runtime_cpp().EPLBAsyncRuntime(device_index)


def warmup_eplb_async_runtime_cpp():
    _load_eplb_async_runtime_cpp()


def create_metadata_field_pair(current: torch.Tensor, next_tensor: torch.Tensor):
    pair = _load_eplb_async_runtime_cpp().MetadataFieldPair()
    pair.current = current
    pair.next = next_tensor
    return pair


def create_reset_tensor_spec(tensor: torch.Tensor, layer_dim: int):
    spec = _load_eplb_async_runtime_cpp().ResetTensorSpec()
    spec.tensor = tensor
    spec.layer_dim = layer_dim
    return spec


def create_layer_plan(
    *,
    layer_id: int,
    routed_experts_weights,
    host_expert_tensors_per_copy,
    dst_slots,
):
    layer_plan = _load_eplb_async_runtime_cpp().LayerPlan()
    layer_plan.layer_id = layer_id
    layer_plan.routed_experts_weights = routed_experts_weights
    layer_plan.host_expert_tensors_per_copy = host_expert_tensors_per_copy
    layer_plan.dst_slots = dst_slots
    return layer_plan


def create_prepared_plan(
    *,
    update_layer_ids,
    layer_plans,
    gpu_metadata_fields,
    cpu_metadata_fields,
    gpu_reset_tensors,
    cpu_reset_tensors,
):
    prepared_plan = _load_eplb_async_runtime_cpp().PreparedPlan()
    prepared_plan.update_layer_ids = update_layer_ids
    prepared_plan.layer_plans = layer_plans
    prepared_plan.gpu_metadata_fields = gpu_metadata_fields
    prepared_plan.cpu_metadata_fields = cpu_metadata_fields
    prepared_plan.gpu_reset_tensors = gpu_reset_tensors
    prepared_plan.cpu_reset_tensors = cpu_reset_tensors
    return prepared_plan
