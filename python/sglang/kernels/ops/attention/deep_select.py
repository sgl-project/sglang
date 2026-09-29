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

# NOTE: see the comments in `entry.cuh`
_INPUT_STRIDE_ALIGNMENT_BYTES = 128
_OUTPUT_STRIDE_ALIGNMENT_BYTES = 32
_CLUSTER_MAX_TOPK = 1024
_CLUSTER_MAX_BATCH_SIZE = 6


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
        page_transform: bool,
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
            page_transform,
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
SUPPORTED_CUDA_ARCHS = ((9, 0), (10, 0), (10, 3))


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
    page_transform: bool,
) -> Module:
    # Validates the call signature; `cache_once` runs this once per valid signature
    if value_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"input dtype must be bfloat16 or float32, got {value_dtype}")
    if index_dtype not in (torch.int32, torch.int64):
        raise ValueError(f"indices_type must be int32 or int64, got {index_dtype}")
    if sorted_value:
        if value_dtype != torch.float32:
            raise ValueError("sorted is only supported for float32 input")
        if not return_value or sorted_index:
            raise ValueError("sorted requires return_value and excludes sorted_index")
    if page_transform:
        if index_dtype != torch.int32 or return_value or sorted_value:
            raise ValueError("the page transform takes int32 indices without values")
    if cluster_size == 1:
        configs = _NORMAL_CONFIGS[(value_dtype, max_topk)]
        host_dispatch = "TopkNormal"
    else:
        # Upstream tunes the cluster kernel for bf16 at max_topk 1024 alone, so
        # the 512 and 1024 buckets share this module.
        if value_dtype != torch.bfloat16 or max_topk != _CLUSTER_MAX_TOPK:
            raise ValueError(
                "the cluster kernel is tuned for bfloat16 at max_topk 1024 only"
            )
        configs = [_Config(max_topk, 256, 1, 4096, 4096, 16, cluster_size)]
        host_dispatch = "TopkCluster"
    classes = [
        config.to_cpp_class_name(
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            sorted_value=sorted_value,
            sorted_index=sorted_index,
            return_value=return_value,
            page_transform=page_transform,
        )
        for config in configs
    ]
    classes = make_cpp_args(*classes)
    root = (KERNEL_PATH / "csrc" / "deep_select" / "vendor").resolve()
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
            page_transform,
        ),
        cuda_files=["deep_select/entry.cuh"],
        cuda_wrappers=[("topk", f"deep_select::{host_dispatch}<{classes}>::topk")],
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


def get_output_stride_alignment_bytes() -> int:
    return _OUTPUT_STRIDE_ALIGNMENT_BYTES


@cache_once
def is_deep_select_supported() -> bool:
    """Return whether DeepSelect JIT supports a CUDA device."""
    if torch.version.cuda is None or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) in SUPPORTED_CUDA_ARCHS


def topk(
    input: torch.Tensor,
    topk: int,
    *,
    sorted: bool = False,
    begin: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int32,
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

    Returns ``(values, indices)``; ``values`` is None when ``return_value`` is
    False, which skips writing them and is about 10% faster.
    """
    assert hint is None, "hint is not supported"
    return _launch_topk_impl(
        input,
        topk,
        sorted=sorted,
        begin=begin,
        end=end,
        indices_type=indices_type,
        sorted_index=sorted_index,
        output_idx=output_idx,
        output_idx_offset=output_idx_offset,
        page_table=None,
        page_size=0,
        idx_oob_fill_value=idx_oob_fill_value,
        value_oob_fill_value=value_oob_fill_value,
        return_value=return_value,
        abort_when_nan_found=abort_when_nan_found,
    )


def topk_page_transform(
    input: torch.Tensor,
    topk: int,
    *,
    page_table: torch.Tensor,
    page_size: int,
    end: Optional[torch.Tensor] = None,
    sorted_index: bool = False,
    output_idx: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = -1,
    abort_when_nan_found: bool = True,
) -> torch.Tensor:
    """Top-k indices of every row of ``input``, written through a page table.

    A selected column ``i`` of row ``r`` is stored as
    ``page_table[r, i // page_size] * page_size + i % page_size``; slots past a
    row's ``end`` hold ``idx_oob_fill_value``. ``page_size`` must be a power of
    2, and ``page_table`` (int32, ``[rows, num_pages]``, unit column stride)
    must cover the whole row: ``num_pages * page_size >= input.shape[1]``.

    This is the narrow case of :func:`topk`: int32 indices, no values, no
    value sort. Returns the int32 ``[rows, topk]`` indices.
    """
    _, indices = _launch_topk_impl(
        input,
        topk,
        sorted=False,
        begin=None,
        end=end,
        indices_type=torch.int32,
        sorted_index=sorted_index,
        output_idx=output_idx,
        output_idx_offset=None,
        page_table=page_table,
        page_size=page_size,
        idx_oob_fill_value=idx_oob_fill_value,
        value_oob_fill_value=0.0,
        return_value=False,
        abort_when_nan_found=abort_when_nan_found,
    )
    return indices


def _launch_topk_impl(
    input: torch.Tensor,
    topk: int,
    *,
    sorted: bool,
    begin: Optional[torch.Tensor],
    end: Optional[torch.Tensor],
    indices_type: torch.dtype,
    sorted_index: bool,
    output_idx: Optional[torch.Tensor],
    output_idx_offset: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
    page_size: int,
    idx_oob_fill_value: int,
    value_oob_fill_value: float,
    return_value: bool,
    abort_when_nan_found: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    rows, vocab_size = input.shape
    device_capability = torch.cuda.get_device_capability(input.device)
    cluster_size, cluster_min_vocab_size = _cluster_tuning(device_capability)
    use_cluster = (
        input.dtype == torch.bfloat16
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
            alignment_bytes=_OUTPUT_STRIDE_ALIGNMENT_BYTES,
        )
    if output_idx is None:
        output_idx = aligned_new_empty(
            (rows, topk),
            indices_type,
            input.device,
            alignment_bytes=_OUTPUT_STRIDE_ALIGNMENT_BYTES,
        )
    module = _jit_deep_select_module(
        input.dtype,
        indices_type,
        sorted,
        sorted_index,
        return_value,
        _get_max_topk_bucket(topk, use_cluster),
        cluster_size if use_cluster else 1,
        page_table is not None,
    )
    input_storage_bytes = (
        input.untyped_storage().nbytes() - input.storage_offset() * input.element_size()
    )
    module.topk(
        input,
        values,
        output_idx,
        begin,
        end,
        output_idx_offset,
        page_table,
        page_size,
        topk,
        input_storage_bytes,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found,
    )
    return values, output_idx
