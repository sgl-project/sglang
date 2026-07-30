from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>

__global__ void sumpad_probe_kernel(
    int tag,
    long long graph_rows,
    long long graph_cols,
    const int* dyn,
    int* counter) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int launch_index = atomicAdd(counter, 1);
    printf(
        "[KPROBE] rank=%d tag=%d launch=%d graph_rows=%lld graph_cols=%lld "
        "host_step=%d real_local_tokens=%d padded_local_tokens=%d "
        "dp_pad_mode=%d dp_buffer_len=%d global_max_tokens=%d global_sum_tokens=%d used_prefill_graph=%d\n",
        dyn[0], tag, launch_index, graph_rows, graph_cols,
        dyn[1], dyn[2], dyn[3], dyn[4], dyn[5], dyn[6], dyn[7], dyn[8]);
  }
}

void sumpad_probe(int64_t tag, at::Tensor x, at::Tensor dyn, at::Tensor counter) {
  TORCH_CHECK(x.is_cuda(), "x must be cuda");
  TORCH_CHECK(dyn.is_cuda() && dyn.scalar_type() == at::kInt, "dyn must be cuda int32");
  TORCH_CHECK(counter.is_cuda() && counter.scalar_type() == at::kInt, "counter must be cuda int32");
  sumpad_probe_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<int>(tag),
      static_cast<long long>(x.size(0)),
      static_cast<long long>(x.dim() > 1 ? x.size(1) : 1),
      dyn.data_ptr<int>(),
      counter.data_ptr<int>());
}
"""

_CPP_SOURCE = r"""
void sumpad_probe(int64_t tag, at::Tensor x, at::Tensor dyn, at::Tensor counter);
"""

_DYN_SLOTS = 9

_module = None
_dyn_gpu: Optional[torch.Tensor] = None
_dyn_cpu: Optional[torch.Tensor] = None
_counter: Optional[torch.Tensor] = None
_step = 0


def build_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="sumpad_shape_probe",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["sumpad_probe"],
            verbose=False,
            build_directory=_build_directory(),
        )
    return _module


def _build_directory() -> str:
    path = os.environ.get(
        "SGLANG_DBG_SHAPE_PROBE_BUILD_DIR", "/tmp/sumpad_shape_probe_build"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_buffers() -> None:
    global _dyn_gpu, _dyn_cpu, _counter
    if _dyn_gpu is None:
        _dyn_gpu = torch.zeros(_DYN_SLOTS, dtype=torch.int32, device="cuda")
        _dyn_cpu = torch.zeros(_DYN_SLOTS, dtype=torch.int32).pin_memory()
        _counter = torch.zeros(1, dtype=torch.int32, device="cuda")


def note_step(
    *,
    rank: int,
    real_local_tokens: int,
    padded_local_tokens: int,
    dp_pad_mode: int,
    dp_buffer_len: int,
    global_max_tokens: int,
    global_sum_tokens: int,
) -> None:
    global _step
    _ensure_buffers()
    _step += 1
    values = [
        rank,
        _step,
        real_local_tokens,
        padded_local_tokens,
        dp_pad_mode,
        dp_buffer_len,
        global_max_tokens,
        global_sum_tokens,
        0,
    ]
    _dyn_cpu.copy_(torch.tensor(values, dtype=torch.int32))
    _dyn_gpu.copy_(_dyn_cpu, non_blocking=True)


def note_used_prefill_graph(used: bool) -> None:
    _ensure_buffers()
    _dyn_cpu[8] = 1 if used else 0
    _dyn_gpu[8].copy_(_dyn_cpu[8], non_blocking=True)


def probe(tag: int, x: torch.Tensor) -> None:
    _ensure_buffers()
    build_module().sumpad_probe(tag, x, _dyn_gpu, _counter)
