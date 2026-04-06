import logging

import torch

from sglang.srt.utils import cpu_has_rvv_support, is_host_cpu_riscv
from sglang.srt.utils.common import use_riscv_rvv_backend

logger = logging.getLogger(__name__)

_convert_weight_packed = None


def _get_convert_weight_packed_op():
    global _convert_weight_packed
    if _convert_weight_packed is not None:
        return _convert_weight_packed

    try:
        import sgl_kernel  # noqa: F401

        _convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
        return _convert_weight_packed
    except (ImportError, AttributeError, RuntimeError):
        if is_host_cpu_riscv():
            logger.warning_once(
                "[RVV] sgl_kernel.convert_weight_packed not found. "
                "Weight packing will be disabled; performance will be degraded. "
                "Ensure sgl-kernel was built with RVV support."
            )
        return None


def _is_lora_wrapped(module) -> bool:
    return hasattr(module, "set_lora") and hasattr(module, "apply_lora")


def _rvv_process_weight_after_loading(module, weight_names) -> None:
    """Pack weights for the RVV backend."""
    devices = {getattr(module, weight_name).device for weight_name in weight_names}
    assert len(devices) == 1, "Expects all weights to be on the same device"
    if devices.pop() != torch.device("cpu"):
        return

    if not cpu_has_rvv_support():
        return

    convert_weight_packed = _get_convert_weight_packed_op()
    if convert_weight_packed is None:
        raise RuntimeError(
            "[RVV] sgl_kernel.convert_weight_packed unavailable; "
            "cannot pack weights for RVV backend. "
            "Rebuild sgl-kernel with RVV support."
        )

    for name in weight_names:
        w = getattr(module, name)
        packed = torch.nn.Parameter(convert_weight_packed(w.data), requires_grad=False)
        packed.__dict__.update(w.__dict__)
        setattr(module, name, packed)

    module.use_riscv_rvv_backend = True

    if getattr(module, "bias", None) is not None and module.bias.dtype != torch.float32:
        module.bias = torch.nn.Parameter(module.bias.float(), requires_grad=False)


class PackRVVWeightMethod:
    def __init__(self, weight_names):
        self.weight_names = weight_names

    def process_weights_after_loading(self, module) -> None:
        _rvv_process_weight_after_loading(module, self.weight_names)


def resolve_rvv_lm_head_weight(lm_head) -> torch.Tensor:
    """Return the weight tensor to pass to the RVV packed-linear kernel."""
    if not hasattr(lm_head, "weight"):
        raise RuntimeError("[RVV] lm_head has no weight tensor.")

    if _is_lora_wrapped(lm_head):
        raise RuntimeError(
            "[RVV] LoRA-wrapped lm_head does not use the RVV lm_head path."
        )

    if use_riscv_rvv_backend(lm_head):
        return lm_head.weight

    weight = lm_head.weight
    if weight.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"[RVV] Unsupported lm_head dtype {weight.dtype} for RVV packed linear."
        )

    convert_weight_packed = _get_convert_weight_packed_op()
    if not cpu_has_rvv_support() or convert_weight_packed is None:
        raise RuntimeError(
            "[RVV] sgl_kernel.convert_weight_packed unavailable; "
            "cannot resolve packed lm_head weight for RVV backend."
        )

    # lm_head is intentionally handled differently from ordinary linear layers.
    # We avoid unconditional load-time prepacking here because:
    # 1. lm_head.weight may be tied to the token embedding table, so replacing it
    #    with a packed layout could corrupt the embedding path.
    # 2. some models wrap lm_head with LoRA-style adapters, which must bypass the
    #    packed RVV path entirely.
    # 3. tests and real workloads may update lm_head.weight in place after load,
    #    so the packed cache must be refreshed lazily when the source changes.
    source_sig = (
        weight.data_ptr(),
        tuple(weight.shape),
        weight.dtype,
        weight._version,
    )
    cached_sig = getattr(lm_head, "_rvv_lm_head_packed_source_sig", None)
    cached_weight = getattr(lm_head, "_rvv_lm_head_packed_weight", None)

    if cached_weight is not None and cached_sig == source_sig:
        return cached_weight

    packed_weight = convert_weight_packed(weight.data)
    setattr(lm_head, "_rvv_lm_head_packed_weight", packed_weight)
    setattr(lm_head, "_rvv_lm_head_packed_source_sig", source_sig)
    return packed_weight


def use_rvv_lm_head_backend(lm_head) -> bool:
    """Whether lm_head is eligible for the RVV packed-linear path."""
    if not hasattr(lm_head, "weight"):
        return False
    if _is_lora_wrapped(lm_head):
        return False
    if lm_head.weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    if use_riscv_rvv_backend(lm_head):
        return True
    return cpu_has_rvv_support() and _get_convert_weight_packed_op() is not None
