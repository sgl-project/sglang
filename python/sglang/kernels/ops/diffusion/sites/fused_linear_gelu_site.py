"""Fused linear + tanh-GELU via the cublasLt GELU epilogue, gated by quality.

Many diffusion DiT FeedForwards compute ``gelu(linear(x), approximate="tanh")``
as a standalone up-projection GEMM followed by a separate, bandwidth-bound GELU
kernel over the ``[tokens, 4*dim]`` MLP intermediate. ``torch._addmm_activation``
folds the bias-add and GELU into the GEMM epilogue (cublasLt), removing the
extra kernel launch and the intermediate HBM round-trip. cublasLt's GELU is the
tanh-approximate GELU (max abs diff ~5e-6 vs ``F.gelu(approximate="tanh")`` in
fp32), so for half-precision inference the fused path differs from the
reference only at bf16/fp16 rounding-order level -- close, but not bit-exact.

Because it is not bit-exact, the fused path is **mounted only for
``quality="high"`` requests** (see ``SamplingParams.quality``): model code marks
its GELU up-projection sites with ``FUSED_LINEAR_GELU.mark`` (default: off,
reference path, bit-exact), and the denoising stage mounts / unmounts
``FUSED_LINEAR_GELU`` at batch boundaries. Mounting is all-or-nothing per transformer: if any marked site
fails the static guards (quantized weights, missing bias, non-half dtype, ...)
no site on that transformer is fused.

The fused GEMM is exposed as a registered custom op (``register_custom_op``)
exactly like the other diffusion kernels (e.g. qknorm_rope), so it stays a
single opaque op under ``torch.compile`` -- no graph break.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch
import torch.nn as nn

from sglang.kernels.jit.utils import get_jit_cuda_arch
from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion
from sglang.srt.utils.custom_op import register_custom_op

# ``torch._addmm_activation`` is the (private but stable) entry point to the
# cublasLt GEMM+bias+activation epilogue. Guard for builds where it is absent
# so the reference path is always available.
_HAS_ADDMM_ACTIVATION = hasattr(torch, "_addmm_activation")


def _fused_linear_gelu_tanh_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@register_custom_op(
    op_name="diffusion_fused_linear_gelu_tanh",
    mutates_args=[],
    fake_impl=_fused_linear_gelu_tanh_fake,
)
def fused_linear_gelu_tanh(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """``gelu_tanh(x @ weight.T + bias)`` fused in the cublasLt GELU epilogue.

    ``weight`` is ``[out, in]`` (nn.Linear / sglang linear layout). Registered
    as a custom op so it is opaque under torch.compile.
    """
    x2d = x.reshape(-1, x.shape[-1])
    out = torch._addmm_activation(bias, x2d, weight.t(), use_gelu=True)
    return out.view(*x.shape[:-1], weight.shape[0])


def _is_unquantized(linear: Any) -> bool:
    """True iff ``linear`` carries plain, unquantized weights."""
    # Quantized checkpoints can leave selected layers unquantized via their
    # exclude list; keep every layer of a quantized model on the reference
    # path (all-or-nothing would reject the model anyway).
    if getattr(linear, "quant_config", None) is not None:
        return False
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None:
        # Plain nn.Linear has no quant_method.
        return True
    from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod

    return isinstance(quant_method, UnquantizedLinearMethod)


def _static_reject_reason(linear: Any) -> str | None:
    """Why ``linear`` may never use the epilogue, or None if it may.

    Input-independent guards: requires the fused API and an unquantized,
    bias'd, non-bias-deferring linear with a half-precision 2D weight. A
    column-parallel layer that gathers across multiple ranks is excluded (the
    per-shard fused op would skip the cross-rank gather);
    ``gather_output=False`` sharded layers are fine because the local shard is
    exactly what the reference forward multiplies. The weight's *device* is
    deliberately not checked here -- under CPU offload the weights live on CPU
    between requests -- the runtime guard checks the input device per call.
    """
    if not _HAS_ADDMM_ACTIVATION:
        return "torch._addmm_activation unavailable"
    if not _is_unquantized(linear):
        return "quantized linear"
    if getattr(linear, "skip_bias_add", False):
        return "skip_bias_add (bias returned separately)"
    if getattr(linear, "gather_output", False) and getattr(linear, "tp_size", 1) > 1:
        return "multi-rank gather_output"
    weight = getattr(linear, "weight", None)
    bias = getattr(linear, "bias", None)
    if weight is None or bias is None or weight.dim() != 2:
        return "missing bias or non-2D weight"
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return f"non-half weight dtype {weight.dtype}"
    if bias.dtype != weight.dtype:
        return f"bias dtype {bias.dtype} != weight dtype {weight.dtype}"
    return None


def can_use_linear_gelu_static(linear: Any) -> bool:
    """Input-independent guards: whether ``linear`` may ever use the epilogue."""
    return _static_reject_reason(linear) is None


# Input-independent verdict per linear, computed on first use; mounting and
# unmounting clear it so a re-mount re-reads the linear.
_VERDICTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _input_independent_ok(linear: Any) -> bool:
    arch = get_jit_cuda_arch()
    if arch.major * 10 + arch.minor >= 120:
        # The cublasLt GELU epilogue selected by current SM120 PyTorch/CUDA
        # builds is slower than GEMM + the native GELU kernel for the FLUX
        # production shape (1, 512, 3072 -> 12288).  Keep quality-gated sites
        # on their existing eager path on RTX 5090; SM90 dispatch is unchanged.
        return False
    return can_use_linear_gelu_static(linear)


def can_use_linear_gelu(linear: Any, x: torch.Tensor) -> bool:
    """Whether ``gelu(linear(x))`` can use the fused cublasLt epilogue now."""
    if not (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    if getattr(linear, "weight", None) is None or x.dtype != linear.weight.dtype:
        return False
    verdict = _VERDICTS.get(linear)
    if verdict is None:
        verdict = _VERDICTS[linear] = _input_independent_ok(linear)
    return verdict


def _site_reject_reason(site: nn.Module) -> str | None:
    linear = getattr(site, FUSED_LINEAR_GELU.metadata(site), None)
    return "missing linear" if linear is None else _static_reject_reason(linear)


class _LinearGeluFusion(QualityGatedFusion):
    def mount(self, root: nn.Module) -> bool:
        _VERDICTS.clear()
        return super().mount(root)

    def unmount(self, root: nn.Module) -> None:
        _VERDICTS.clear()
        super().unmount(root)


# A site's marker metadata names its up-projection linear, whose output feeds
# ``F.gelu(..., approximate="tanh")``.
FUSED_LINEAR_GELU = _LinearGeluFusion(
    name="fused linear+GELU",
    marker_attr="_sgl_fused_gelu_linear_attr",
    enabled_attr="_sgl_fused_gelu_enabled",
    reject_reason=_site_reject_reason,
)
