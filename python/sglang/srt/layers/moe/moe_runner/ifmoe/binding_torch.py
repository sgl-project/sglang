"""JIT-compile and load the IFMoe kernel as a PyTorch C++ extension."""

import logging
import os
import threading

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_module = None


def _num_local_experts() -> int:
    raw = os.environ.get("IFMOE_NUM_LOCAL_EXPERTS") or os.environ.get(
        "FUSEMOE_NUM_LOCAL_EXPERTS", "32"
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid IFMOE_NUM_LOCAL_EXPERTS={raw!r}") from exc
    if value not in (32, 64):
        raise ValueError("IFMOE_NUM_LOCAL_EXPERTS must be 32 or 64")
    return value


def _get_cuda_arch_flags():
    """Detect GPU arch and return nvcc flags."""
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    return [f"-gencode=arch=compute_{sm},code=sm_{sm}"]


def get_module_no_warmup():
    """Thread-safe JIT compilation only (no workspace pre-warm)."""
    global _module
    if _module is not None:
        return _module
    with _lock:
        if _module is not None:
            return _module

        kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")
        kernel_cu = os.path.join(kernel_dir, "kernel.cu")
        assert os.path.isfile(kernel_cu), f"kernel.cu not found at {kernel_cu}"

        arch_flags = _get_cuda_arch_flags()
        local_experts = _num_local_experts()
        logger.info(
            f"JIT-compiling IFMoe kernel (arch: {arch_flags}, "
            f"local_experts={local_experts})..."
        )

        _module = load(
            name=f"ifmoe_kernel_el{local_experts}",
            sources=[kernel_cu],
            extra_cuda_cflags=[
                "-DTORCH_BINDING",
                f"-DIFMOE_NUM_LOCAL_EXPERTS={local_experts}",
                "-O2",
                "--use_fast_math",
                "-std=c++17",
                *arch_flags,
            ],
            extra_cflags=["-O2", "-std=c++17"],
            verbose=False,
        )
        logger.info("IFMoe kernel compiled and loaded (no warmup).")
        return _module


def get_module():
    """Thread-safe JIT compilation and loading of the kernel .so."""
    global _module
    if _module is not None:
        return _module
    with _lock:
        if _module is not None:
            return _module

        kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")
        kernel_cu = os.path.join(kernel_dir, "kernel.cu")
        assert os.path.isfile(kernel_cu), f"kernel.cu not found at {kernel_cu}"

        arch_flags = _get_cuda_arch_flags()
        local_experts = _num_local_experts()
        logger.info(
            f"JIT-compiling IFMoe kernel (arch: {arch_flags}, "
            f"local_experts={local_experts})..."
        )

        _module = load(
            name=f"ifmoe_kernel_el{local_experts}",
            sources=[kernel_cu],
            extra_cuda_cflags=[
                "-DTORCH_BINDING",
                f"-DIFMOE_NUM_LOCAL_EXPERTS={local_experts}",
                "-O2",
                "--use_fast_math",
                "-std=c++17",
                *arch_flags,
            ],
            extra_cflags=["-O2", "-std=c++17"],
            verbose=False,
        )
        logger.info("IFMoe kernel compiled and loaded.")

        # Pre-warm with T=1 dummy data to trigger workspace allocation
        # while GPU has maximum free memory (before model weights are loaded).
        try:
            import torch

            dev = torch.cuda.current_device()
            d = f"cuda:{dev}"
            if local_experts == 32:
                H, I, EL = 7168, 2048, local_experts
                _module.kernel(
                    torch.zeros(1, 256, dtype=torch.float32, device=d),
                    torch.zeros(256, dtype=torch.bfloat16, device=d),
                    torch.zeros(1, H, dtype=torch.float8_e4m3fn, device=d),
                    torch.ones(H // 128, 1, dtype=torch.float32, device=d),
                    torch.zeros(EL, 2 * I, H, dtype=torch.float8_e4m3fn, device=d),
                    torch.ones(
                        EL, 2 * I // 128, H // 128, dtype=torch.float32, device=d
                    ),
                    torch.zeros(EL, H, I, dtype=torch.float8_e4m3fn, device=d),
                    torch.ones(EL, H // 128, I // 128, dtype=torch.float32, device=d),
                    0,
                    1.0,
                    torch.empty(0, dtype=torch.int32, device=d),
                    torch.empty(0, dtype=torch.float32, device=d),
                )
                torch.cuda.synchronize()
                logger.info("IFMoe workspace pre-warmed successfully.")
            else:
                logger.info("Skipping IFMoe dummy pre-warm for local_experts=64.")
        except Exception as e:
            logger.warning(f"IFMoe workspace pre-warm failed: {e}")
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        return _module


def kernel(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
    ext_topk_ids=None,
    ext_topk_weights=None,
):
    """Call the IFMoe kernel with PyTorch tensors."""
    import torch

    mod = get_module()
    if ext_topk_ids is None:
        ext_topk_ids = torch.empty(0, dtype=torch.int32, device=routing_logits.device)
    if ext_topk_weights is None:
        ext_topk_weights = torch.empty(
            0, dtype=torch.float32, device=routing_logits.device
        )
    return mod.kernel(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        local_expert_offset,
        routed_scaling_factor,
        ext_topk_ids,
        ext_topk_weights,
    )
