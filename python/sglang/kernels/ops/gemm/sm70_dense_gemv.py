"""Native FP16 projections for the measured Qwen3.8 TP4 and GLM-5.3-Flash TP8 shapes."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

# (output features, input features) -> (threads, lanes per row, vector width).
_CONFIGS = {
    (16384, 2560): (64, 32, 8),  # TP4 shard of the 65,536-token MTP hot head
    (4096, 2560): (64, 16, 8),
    (3584, 2560): (64, 32, 8),
    (2560, 1536): (128, 32, 8),
    (320, 2560): (256, 32, 8),
    (2560, 160): (128, 8, 4),
    (512, 2560): (256, 32, 8),
    (640, 2560): (128, 32, 8),
    (24, 2560): (64, 32, 8),
    (1, 2560): (256, 32, 8),
    (10240, 2560): (64, 32, 8),
    (2560, 2560): (128, 32, 8),
    # GLM-5.3-Flash TP8 attention: KDA qkv+b+f_a+g_a and o_proj, DSA o_proj,
    # indexer wq_b, q_a+kv_a and q_b. Swept on V100 against cuBLAS at M=1.
    (3336, 4096): (256, 16, 8),
    (4096, 1024): (64, 16, 8),
    (4096, 2048): (128, 16, 8),
    (4096, 1536): (64, 16, 8),
    (2048, 4096): (64, 32, 8),
    (2048, 1536): (128, 16, 8),
}

# GLM verify and draft-extend rows (2-4) reuse the one-row config through the
# small GEMM, so each row is bitwise the decode GEMV.
_ROW_EXACT = {
    (3336, 4096),
    (4096, 1024),
    (4096, 2048),
    (4096, 1536),
    (2048, 4096),
    (2048, 1536),
}


def supported(x: torch.Tensor, weight: torch.Tensor, bias=None) -> bool:
    from sglang.kernels.ops.gemm.sm70_small_gemm import shape_supported

    return (
        envs.SGLANG_SM70_DENSE_GEMV.get()
        and x.is_cuda
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and weight.device == x.device
        and x.ndim == 2
        and x.shape[0] in (1, 2, 3, 4)
        and weight.ndim == 2
        and x.shape[1] == weight.shape[1]
        and bias is None
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and (
            (
                x.shape[0] == 1
                and (
                    tuple(weight.shape) in _CONFIGS
                    or (weight.shape[1] == 2560 and weight.shape[0] >= 32768)
                )
            )
            or tuple(weight.shape) in _ROW_EXACT
            or shape_supported(x, weight)
        )
        and torch.cuda.get_device_capability(x.device) == (7, 0)
    )


@cache_once
def _module(threads: int, lanes: int, vector: int):
    return load_jit(
        "sm70_dense_gemv",
        threads,
        lanes,
        vector,
        cuda_files=["elementwise/sm70_dense_gemv.cuh"],
        cuda_wrappers=[
            ("gemv", f"sglang::sm70_dense_gemv::gemv<{threads},{lanes},{vector}>")
        ],
    )


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.shape[0] != 1:
        from sglang.kernels.ops.gemm import sm70_small_gemm

        if tuple(weight.shape) in _ROW_EXACT:
            threads, lanes, _ = _CONFIGS[tuple(weight.shape)]
            return sm70_small_gemm.linear_with(x, weight, threads, lanes)
        return sm70_small_gemm.linear(x, weight)
    config = _CONFIGS.get(tuple(weight.shape), (64, 32, 8))
    out = torch.empty((1, weight.shape[0]), dtype=x.dtype, device=x.device)
    _module(*config).gemv(x, weight, out)
    return out
