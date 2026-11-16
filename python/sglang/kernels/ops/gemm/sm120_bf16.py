"""BF16 NT Linear backends and first-use tuning on SM120.

Both backends consume contiguous BF16 ``x[M, K]`` and ``weight[N, K]``
and return BF16 ``x @ weight.T`` with FP32 accumulation, without bias.
Warm the selected backend outside CUDA graph capture before capturing it.
Backend performance policy belongs to the caller.
"""

from collections.abc import Callable
from functools import cache

import torch


def linear_bf16_quack(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from quack.gemm_interface import gemm_out

    out = x.new_empty((x.shape[0], weight.shape[0]))
    gemm_out(x, weight.T, out, tuned=True, split_k=None)
    return out


@cache
def _cublaslt_runner_and_config():
    from flashinfer.autotuner import TuningConfig
    from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module

    # cuBLASLt enumerates algorithms per exact shape. An algorithm's index
    # cannot be reused at a different M, even within the same tuning bucket.
    return (
        get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner(),
        TuningConfig(
            use_cuda_graph=True,
            use_cold_l2_graph_replay=True,
            use_cold_l2_cache=True,
            cold_chain_launches=8,
            profile_arena_input_indices=(1,),
        ),
    )


def linear_bf16_cublaslt(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from flashinfer.autotuner import AutoTuner

    out = x.new_empty((x.shape[0], weight.shape[0]))
    # mm_bf16's process-wide scratch buffer is unsafe for concurrent streams
    # or graphs. The allocator owns this call's scratch, including during capture.
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=x.device)
    inputs = [x, weight.T, None, False, out, workspace]
    runner, config = _cublaslt_runner_and_config()
    runner, tactic = AutoTuner.get().choose_one(
        "sglang_bf16_cublaslt", [runner], config, inputs
    )
    runner(inputs=inputs, tactic=tactic)
    return out


@cache
def _is_sm120(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (12, 0)


class Sm120Bf16Linear:
    """Execute a caller-selected backend, or return None for native fallback.

    Accepts nonempty, contiguous BF16 operands with N and K divisible by 8,
    without bias or autograd. M is not restricted to a model's measured shapes.
    Only backend/shape/device warm metadata is retained. Backend tactics belong
    to the libraries, and temporary GPU scratch belongs to the PyTorch allocator.
    """

    def __init__(self) -> None:
        self._warmed = set()

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        backend: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor | None:
        if backend not in (linear_bf16_quack, linear_bf16_cublaslt):
            raise ValueError("Expected the QuACK or cuBLASLt BF16 Linear backend")
        if (
            torch.compiler.is_compiling()
            or torch.version.hip
            or bias is not None
            or x.device.type != "cuda"
            or weight.device != x.device
            or x.dtype != torch.bfloat16
            or weight.dtype != x.dtype
            or x.ndim < 2
            or weight.ndim != 2
            or not x.is_contiguous()
            or not weight.is_contiguous()
            or x.shape[-1] != weight.shape[1]
            or x.numel() == 0
            or weight.numel() == 0
            or weight.shape[0] % 8
            or weight.shape[1] % 8
            or (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
            or x.data_ptr() % 16
            or weight.data_ptr() % 16
            or not _is_sm120(x.device.index)
        ):
            return None
        key = (backend, x.device.index, x.numel() // x.shape[-1], *weight.shape)
        cold = key not in self._warmed
        with torch.cuda.device(x.device):
            if cold and torch.cuda.is_current_stream_capturing():
                return None
            if cold and backend is linear_bf16_cublaslt:
                from flashinfer.autotuner import autotune

                # Tuning mode scans profiles even on cache hits.
                with autotune(True):
                    out = backend(x.view(-1, x.shape[-1]), weight)
            else:
                out = backend(x.view(-1, x.shape[-1]), weight)
            if cold:
                self._warmed.add(key)
        return out.view(*x.shape[:-1], weight.shape[0])
