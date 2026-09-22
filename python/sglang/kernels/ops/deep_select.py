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
def _cluster_tuning() -> Tuple[int, int]:
    # Return (cluster_size, min_vocab_size)
    major = torch.cuda.get_device_capability()[0]
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
    assert 9 <= torch.cuda.get_device_capability()[0] <= 10
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
        extra_cuda_cflags=["--expt-extended-lambda", "--use_fast_math", "--ftz=false"],
    )


def _get_max_topk_bucket(topk: int, use_cluster: bool) -> int:
    if use_cluster:
        return _CLUSTER_MAX_TOPK
    return 512 if topk <= 512 else 1024 if topk <= 1024 else 4096


def get_input_stride_alignment_bytes() -> int:
    return _INPUT_STRIDE_ALIGNMENT_BYTES


def topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int32,
    sorted_index: bool = False,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Select the largest ``topk`` values of every row of ``input``.

    Follows the public DeepSelect interface, minus ``begin`` and ``hint``, which
    upstream does not implement. ``input`` must be bfloat16 or float32 with a
    row stride that is a multiple of ``_INPUT_STRIDE_ALIGNMENT_BYTES``; ``end``
    is the per-row exclusive valid length. Neither output is sorted unless
    ``sorted`` (by value, float32 only) or ``sorted_index`` asks for it, and
    both are allocated with a padded row stride, so they may not be contiguous.

    Returns ``(values, indices)``; ``values`` is None when ``return_value`` is
    False, which skips writing them and is about 10% faster.
    """
    rows, vocab_size = input.shape
    cluster_size, cluster_min_vocab_size = _cluster_tuning()
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
    module = _jit_deep_select_module(
        input.dtype,
        indices_type,
        sorted,
        sorted_index,
        return_value,
        _get_max_topk_bucket(topk, use_cluster),
        cluster_size if use_cluster else 1,
    )
    module.topk(
        input,
        values,
        output_idx,
        end,
        output_idx_offset,
        topk,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found,
    )
    return values, output_idx
