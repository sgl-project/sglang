"""Cake frozen MegaMoE top-k reducer (``partials [cap, 6, 4096] -> bf16 [cap, 4096]``) via FlashInfer.

FlashInfer entry: ``flashinfer.jit.cake_megamoe_topk_reduce.run_cake_megamoe_topk_reduce(partials,
out, num_tokens)`` (module loader ``get_cake_megamoe_topk_reduce_module(device)``,
``load_cake_megamoe_topk_reduce_module(arch)``, ``supported_capabilities()``).
In FlashInfer the reducer is the terminal stage of the SM100 CuTe-DSL NVFP4
megakernel backend (``Nvfp4CutedslMegaKernelBackend._uses_native_topk_reduce``
selects it automatically for ``max_tokens_per_rank in {256, 4096}``,
``token_hidden_size == 4096``, ``top_k == 6``, ``combine_dtype="bf16"``,
``apply_topk_in_fc1=True``, ``enable_in_kernel_fc2_reduce=False``); only the
reducer is Cake-generated. Contract at FlashInfer ``46340689a5ab``
(``cake_megamoe_topk_reduce_binding.cuh``): exact CC (10,0) / (10,3); BF16
``partials [capacity, 6, 4096]`` with ``capacity in {256, 4096}``; BF16
``out [capacity, 4096]``; ``0 <= num_tokens <= capacity``; 16-byte aligned
pointers; FP32 accumulation over the 6 routed partials, one BF16 rounding;
grid ``4 * num_tokens`` x 256 threads, no dynamic smem.

CUDA graphs: the module must be loaded BEFORE capture (``MoEEpLayer.warmup``
in FlashInfer; here call :func:`load_megamoe_topk_reduce_module` once); the
launch itself is one same-stream kernel with no lazy compile / allocation.

Not supported here: other hidden sizes / top-k / capacities, non-BF16 partials,
SM120 / SM121 (fall back to the CuTe-DSL reducer there).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.cake_kernels._support import SM100, SM103, cuda_tensor_on
from sglang.kernels.cake_kernels.moe_common import contiguous_cuda, modules_available

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.jit.cake_megamoe_topk_reduce"
FI_JIT_MODULE = "flashinfer.jit.cake_megamoe_topk_reduce"
ARCHS = (SM100, SM103)
HIDDEN = 4096
TOP_K = 6
CAPACITIES = (256, 4096)
ALIGNMENT = 16


def supports_megamoe_topk_reduce(
    partials: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
) -> bool:
    """Admission check mirroring the binding's ``TVM_FFI_ICHECK``s; never raises."""
    import torch

    try:
        if not (modules_available(FI_MODULE) and cuda_tensor_on(partials, ARCHS)):
            return False
        if partials.ndim != 3:
            return False
        capacity = int(partials.shape[0])
        return (
            capacity in CAPACITIES
            and contiguous_cuda(
                partials, shape=(capacity, TOP_K, HIDDEN), dtype=torch.bfloat16
            )
            and contiguous_cuda(out, shape=(capacity, HIDDEN), dtype=torch.bfloat16)
            and out.device == partials.device
            and 0 <= int(num_tokens) <= capacity
            and partials.data_ptr() % ALIGNMENT == 0
            and out.data_ptr() % ALIGNMENT == 0
        )
    except Exception:
        return False


def load_megamoe_topk_reduce_module(device: Optional[torch.device] = None):
    """Load (JIT-build if needed) the frozen reducer for ``device``; call before graph capture."""
    from flashinfer.jit.cake_megamoe_topk_reduce import (
        get_cake_megamoe_topk_reduce_module,
    )

    return get_cake_megamoe_topk_reduce_module(device)


def megamoe_topk_reduce(
    partials: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Sum the 6 BF16 partials of the first ``num_tokens`` rows into ``out`` (FP32 accumulate); returns ``out``."""
    from flashinfer.jit.cake_megamoe_topk_reduce import run_cake_megamoe_topk_reduce

    run_cake_megamoe_topk_reduce(partials, out, num_tokens)
    return out
