from __future__ import annotations

from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    KERNEL_PATH,
    aligned_new_empty,
    cache_once,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Must match INPUT_/OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT in vendor/structs.h; the
# kernel's TMA descriptor depends on them and `entry.cuh` rejects a row that
# misses one.
_INPUT_STRIDE_ALIGNMENT_BYTES = 1024
_OUTPUT_STRIDE_ALIGNMENT_BYTES = 32
_CLUSTER_MAX_TOPK = 1024
_CLUSTER_MAX_BATCH_SIZE = 6
_SUPPORTED_CAPABILITIES = ((9, 0), (10, 0), (10, 3))


class _Config(NamedTuple):
    max_topk: int
    num_threads: int
    occupancy: int
    elements_per_round: int
    reconstruct_threshold: int
    tma_buffer_depth: int
    cluster_size: int = 1

    def to_cpp_class_name(
        self,
        value_dtype: torch.dtype,
        index_dtype: torch.dtype,
        sorted_value: bool,
        sorted_index: bool,
        return_value: bool,
    ) -> str:
        template_args = make_cpp_args(
            value_dtype,
            index_dtype,
            sorted_value,
            sorted_index,
            return_value,
            self.max_topk,
            self.num_threads,
            self.occupancy,
            self.elements_per_round,
            self.reconstruct_threshold,
            self.tma_buffer_depth,
            512,  # elements_per_segment, fixed upstream
            self.cluster_size,
        )
        return f"TopkSelectConfig<{template_args}>"


# Per (value dtype, top-k bucket): the tuning for a single-wave batch and, where
# the bucket splits on wave count, the tuning for anything larger. A single
# entry means no split, and then only one kernel is instantiated.
_NORMAL_CONFIGS: Dict[Tuple[torch.dtype, int], Tuple[_Config, ...]] = {
    (torch.bfloat16, 512): (
        _Config(512, 512, 1, 8192, 4096, 5),
        _Config(512, 256, 2, 4096, 4096, 4),
    ),
    (torch.bfloat16, 1024): (
        _Config(1024, 512, 1, 8192, 4096, 5),
        _Config(1024, 256, 2, 4096, 4096, 3),
    ),
    (torch.bfloat16, 4096): (_Config(4096, 512, 1, 8192, 4096, 3),),
    (torch.float32, 512): (_Config(512, 512, 1, 8192, 4096, 3),),
    (torch.float32, 1024): (_Config(1024, 512, 1, 8192, 4096, 3),),
    (torch.float32, 4096): (_Config(4096, 256, 1, 4096, 4096, 3),),
}


@cache_once
def _cluster_tuning(device_capability: Tuple[int, int]) -> Tuple[int, int]:
    # Return (cluster_size, min_vocab_size)
    major = device_capability[0]
    return (8, 128 * 1024) if major == 9 else (16, 512 * 1024)


@cache_once
def _jit_deep_select_module(
    value_dtype: torch.dtype,
    index_dtype: torch.dtype,
    sorted_value: bool,
    sorted_index: bool,
    return_value: bool,
    max_topk: int,
    cluster_size: int,
) -> Module:
    assert value_dtype in (torch.bfloat16, torch.float32)
    assert index_dtype in (torch.int32, torch.int64)
    if sorted_value:
        assert value_dtype == torch.float32
        assert return_value and not sorted_index
    if cluster_size == 1:
        configs = _NORMAL_CONFIGS[(value_dtype, max_topk)]
        host_dispatch = "TopkNormal"
    else:
        # Upstream tunes the cluster kernel for bf16 at max_topk 1024 alone, so
        # the 512 and 1024 buckets share this module.
        assert value_dtype == torch.bfloat16
        assert max_topk == _CLUSTER_MAX_TOPK
        configs = [_Config(max_topk, 256, 1, 4096, 4096, 16, cluster_size)]
        host_dispatch = "TopkCluster"
    classes = [
        config.to_cpp_class_name(
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            sorted_value=sorted_value,
            sorted_index=sorted_index,
            return_value=return_value,
        )
        for config in configs
    ]
    classes = make_cpp_args(*classes)
    root = (KERNEL_PATH / "csrc" / "deepselect" / "vendor").resolve()
    return load_jit(
        "deep_select_topk",
        # cache only distinct key for better readability
        *make_cpp_args(
            value_dtype,
            index_dtype,
            sorted_value,
            sorted_index,
            return_value,
            max_topk,
            cluster_size,
        ),
        cuda_files=["deepselect/entry.cuh"],
        cuda_wrappers=[("topk", f"deepselect::{host_dispatch}<{classes}>::topk")],
        extra_include_paths=[
            str(root),
            str(root / "3rdparty" / "kerutils" / "include"),
        ],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=[
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ftz=false",
            "-Xptxas=--register-usage-level=10",
        ],
    )


def _get_max_topk_bucket(topk: int, use_cluster: bool) -> int:
    if use_cluster:
        return _CLUSTER_MAX_TOPK
    return 512 if topk <= 512 else 1024 if topk <= 1024 else 4096


def get_input_stride_alignment_bytes() -> int:
    return _INPUT_STRIDE_ALIGNMENT_BYTES


def get_stride_requirement() -> Tuple[int, int]:
    """Return the input and output row-stride requirements in bytes."""
    return _INPUT_STRIDE_ALIGNMENT_BYTES, _OUTPUT_STRIDE_ALIGNMENT_BYTES


def get_deepselect_supported_architectures() -> Tuple[int, ...]:
    """Return the CUDA compute capabilities supported by the JIT kernel."""
    return tuple(major * 10 + minor for major, minor in _SUPPORTED_CAPABILITIES)


def is_deepselect_supported(device=None) -> bool:
    """Return whether DeepSelect JIT supports a CUDA device."""
    if torch.version.cuda is None or not torch.cuda.is_available():
        return False
    try:
        normalized_device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else (
                torch.device("cuda", torch.cuda.current_device())
                if device is None
                else torch.device(device)
            )
        )
        if normalized_device.type != "cuda":
            return False
        return (
            torch.cuda.get_device_capability(normalized_device)
            in _SUPPORTED_CAPABILITIES
        )
    except (RuntimeError, TypeError, ValueError):
        return False


def _needs_output_staging(output: torch.Tensor, topk: int) -> bool:
    return (
        output.data_ptr() % _OUTPUT_STRIDE_ALIGNMENT_BYTES != 0
        or topk * output.element_size() % _OUTPUT_STRIDE_ALIGNMENT_BYTES != 0
    )


def topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    hint: Optional[torch.Tensor] = None,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Select the largest ``topk`` values of every row of ``input``.

    This follows the public DeepSelect interface. ``end`` is the per-row
    exclusive valid length; ``begin`` and ``hint`` are reserved but unsupported.

    Returns ``(values, indices)``; ``values`` is None when ``return_value`` is
    False, which skips writing them and is about 10% faster.
    """
    if hint is not None:
        raise ValueError("hint is not supported currently")
    if input.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("input dtype must be bfloat16 or float32")
    if indices_type not in (torch.int32, torch.int64):
        raise RuntimeError("indices_type must be int32 or int64")
    if output_idx is not None and output_idx.dtype != indices_type:
        raise ValueError("output_idx dtype must match indices_type")
    if input.device.type != "cuda":
        raise RuntimeError("input must be a CUDA tensor")
    if input.ndim != 2:
        raise RuntimeError("input must be a 2D tensor")
    if not 0 < topk <= 4096:
        raise RuntimeError(f"topk must be in [1, 4096], got {topk}")
    if sorted and not return_value:
        raise RuntimeError("return_value must be enabled when sorted is True")
    if sorted and sorted_index:
        raise RuntimeError("sorted and sorted_index cannot both be True")
    if sorted and input.dtype is torch.bfloat16:
        raise RuntimeError("sorted is only supported for float32 input")

    rows, vocab_size = input.shape
    device_capability = torch.cuda.get_device_capability(input.device)
    if device_capability not in _SUPPORTED_CAPABILITIES:
        major, minor = device_capability
        raise RuntimeError(f"DeepSelect does not support SM{major}{minor}")
    cluster_size, cluster_min_vocab_size = _cluster_tuning(device_capability)
    use_cluster = (
        input.dtype is torch.bfloat16
        and rows <= _CLUSTER_MAX_BATCH_SIZE
        and vocab_size >= cluster_min_vocab_size
        and topk <= _CLUSTER_MAX_TOPK
    )
    values = None
    if return_value:
        values = aligned_new_empty(
            (rows, topk),
            input.dtype,
            input.device,
            alignment=_OUTPUT_STRIDE_ALIGNMENT_BYTES,
        )
    if output_idx is None:
        output_idx = aligned_new_empty(
            (rows, topk),
            indices_type,
            input.device,
            alignment=_OUTPUT_STRIDE_ALIGNMENT_BYTES,
        )
        kernel_output_idx = output_idx
    else:
        kernel_output_idx = (
            aligned_new_empty(
                (rows, topk),
                indices_type,
                input.device,
                alignment=_OUTPUT_STRIDE_ALIGNMENT_BYTES,
            )
            if _needs_output_staging(output_idx, topk)
            else output_idx
        )
    module = _jit_deep_select_module(
        input.dtype,
        indices_type,
        sorted,
        sorted_index,
        return_value,
        _get_max_topk_bucket(topk, use_cluster),
        cluster_size if use_cluster else 1,
    )
    input_storage_bytes = (
        input.untyped_storage().nbytes() - input.storage_offset() * input.element_size()
    )
    module.topk(
        input,
        values,
        output_idx,
        kernel_output_idx,
        begin,
        end,
        output_idx_offset,
        topk,
        input_storage_bytes,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found,
    )
    if kernel_output_idx is not output_idx:
        output_idx.copy_(kernel_output_idx)
    return values, output_idx


deepselect_topk = topk
