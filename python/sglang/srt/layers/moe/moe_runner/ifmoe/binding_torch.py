"""IFMoe kernel binding — dispatches to the AOT-compiled sgl_kernel op.

The kernel CUDA source now lives in ``sgl-kernel/csrc/moe/ifmoe/`` and is
compiled by sgl-kernel's CMake build (SM90 binary only). At runtime we
resolve ``torch.ops.sgl_kernel.ifmoe_kernel`` and refuse to run on SM100+
GPUs, which carry the ``common_ops_sm100`` binary that omits this op.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_kernel_fn = None


def _check_arch() -> None:
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    if sm >= 100:
        raise RuntimeError(
            f"IFMOE only supports Hopper (SM90); detected SM{sm}. "
            "Use --moe-runner-backend flashinfer_trtllm or deep_gemm on "
            "Blackwell/SM100+ hardware. SM100 support is out of scope for this PR."
        )
    if sm != 90:
        raise RuntimeError(
            f"IFMOE requires SM90 (Hopper); detected SM{sm}. "
            "Pre-Hopper architectures are not supported."
        )


def _resolve_kernel():
    """Return the AOT-bound ``sgl_kernel::ifmoe_kernel`` torch op.

    The op is only present in the SM90 sgl-kernel binary; on SM100+ wheels
    the registration is compiled out and the attribute will be missing.
    """
    global _kernel_fn
    if _kernel_fn is not None:
        return _kernel_fn
    _check_arch()
    # Importing sgl_kernel ensures the TORCH_LIBRARY_FRAGMENT is loaded so the
    # ``sgl_kernel::ifmoe_kernel`` op is registered.
    import sgl_kernel  # noqa: F401

    try:
        _kernel_fn = torch.ops.sgl_kernel.ifmoe_kernel
    except AttributeError as exc:  # pragma: no cover - guarded by arch check
        raise RuntimeError(
            "sgl_kernel does not expose ifmoe_kernel — installed wheel was "
            "built without the IFMOE op (SM90 binary required)."
        ) from exc
    return _kernel_fn


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
    fn = _resolve_kernel()
    if ext_topk_ids is None:
        ext_topk_ids = torch.empty(0, dtype=torch.int32, device=routing_logits.device)
    if ext_topk_weights is None:
        ext_topk_weights = torch.empty(
            0, dtype=torch.float32, device=routing_logits.device
        )
    return fn(
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
