"""Cake Kimi-K3 vision tower (MoonViT-3D + PatchMergerV2) via FlashInfer.

FlashInfer entries: ``flashinfer.kimi_k3_vision.kimi_k3_vision_tower`` and
``flashinfer.kimi_k3_vision.prepare_kimi_k3_vision_tower`` (experimental API;
implementation in ``flashinfer.experimental.kimi_k3_vision_tower.cake_backend``,
JIT in ``...cake_jit``; FlashInfer ``46340689a5ab``, flashinfer-ai/flashinfer#4568).

Contract: BF16 contiguous, 16-byte aligned ``pixel_values [T, 3, 14, 14]``
packed in ``grid_thws`` order (t slow, then y, then x); ``grid_thws`` is a host
list of ``(t, h, w)`` with ``1 <= t <= 4``, ``h, w`` even and ``<= 512``,
``T = sum(t * h * w)``; ``weights`` is either the BF16 ``nn.Linear``-style dict
(``patch_proj [1024, 588]``, ``pos_emb [64, 64, 1024]``, ``time_weight [4, 1024]``,
``final_norm [1024]``, ``merger_proj0 [4096, 4096]``, ``merger_proj1 [7168, 4096]``,
``post_norm [7168]``, ``layers`` = list of ``{norm0 [1024], wqkv [4608, 1024],
wo [1024, 1536], norm1 [1024], fc0 [4096, 1024], fc1 [1024, 4096]}``; the
production tower has 27 layers, shorter stacks are accepted) or the
``PreparedWeights`` returned by ``prepare_kimi_k3_vision_weights``. Output BF16
``out [N, 7168]`` with ``N = sum((h / 2) * (w / 2))``.

* ``kimi_k3_vision_tower`` -- one-shot: prepares dict weights on every call
  (use ``prepare_kimi_k3_vision_weights`` once per model) and derives the host
  plan per ``grid_thws`` batch (pass a cached ``plan=`` / ``pos_rows=`` to skip
  it); allocates when ``out`` is omitted.
* ``prepare_kimi_k3_vision_tower`` -> ``KimiK3VisionTowerRunner`` -- every
  allocation at prepare; ``runner.launch()`` writes ``out`` with no allocation
  and no host sync (CUDA-graph capturable, bit-identical to eager). Re-prepare
  when ``grid_thws`` or any tensor binding changes.
* ``prepare_kimi_k3_vision_weights`` -- once per model; patch-projection
  padding + contiguous copies, RMSNorm weights applied activation-side.

Built for exact compute capability 10.0 / 10.3 (sm_100a / sm_103a); the
generated program must register every kernel key of the arch
(``cake_backend.generated_program_available``). PDL is on by default.

Not supported here (keep the existing SGLang path): other ViT configurations,
odd ``h`` / ``w``, ``t > 4``, ``h`` or ``w`` above 512, non-BF16 weights,
SM90 / SM120.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from sglang.kernels.cake_kernels._support import (
    SM100,
    SM103,
    cuda_tensor_on,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.kimi_k3_vision"
FI_BACKEND_MODULE = "flashinfer.experimental.kimi_k3_vision_tower.cake_backend"
FI_JIT_MODULE = "flashinfer.experimental.kimi_k3_vision_tower.cake_jit"
ARCHS = (SM100, SM103)

PATCH = 14
IN_CH = 3
PATCH_DIM = IN_CH * PATCH * PATCH  # 588
HIDDEN = 1024
TEXT_HIDDEN = 7168
POS_EMB_T = 4
ROPE_MAX_HW = 512
WEIGHT_KEYS = (
    "patch_proj",
    "pos_emb",
    "time_weight",
    "final_norm",
    "merger_proj0",
    "merger_proj1",
    "post_norm",
    "layers",
)
LAYER_WEIGHT_KEYS = ("norm0", "wqkv", "wo", "norm1", "fc0", "fc1")


def _grid_thws_ok(grid_thws: Sequence[Sequence[int]]) -> Optional[int]:
    """Total patch count when ``grid_thws`` is admissible, else ``None``."""
    total = 0
    if len(grid_thws) == 0:
        return None
    for item in grid_thws:
        if len(item) != 3:
            return None
        t, h, w = (int(x) for x in item)
        if not (1 <= t <= POS_EMB_T):
            return None
        if h <= 0 or w <= 0 or h % 2 or w % 2 or h > ROPE_MAX_HW or w > ROPE_MAX_HW:
            return None
        total += t * h * w
    return total


def _weights_ok(weights: Any, device) -> bool:
    import torch

    if not isinstance(weights, dict):
        # PreparedWeights (frozen dataclass) from prepare_kimi_k3_vision_weights.
        return hasattr(weights, "layers") and hasattr(weights, "patch_proj")
    if any(key not in weights for key in WEIGHT_KEYS):
        return False
    layers = weights["layers"]
    if not isinstance(layers, (list, tuple)) or not layers:
        return False
    for lw in layers:
        if not isinstance(lw, dict) or any(k not in lw for k in LAYER_WEIGHT_KEYS):
            return False
        if not all(
            isinstance(lw[k], torch.Tensor)
            and lw[k].dtype == torch.bfloat16
            and lw[k].device == device
            for k in LAYER_WEIGHT_KEYS
        ):
            return False
    return all(
        isinstance(weights[k], torch.Tensor)
        and weights[k].dtype == torch.bfloat16
        and weights[k].device == device
        for k in WEIGHT_KEYS
        if k != "layers"
    )


def supports_kimi_k3_vision_tower(
    pixel_values: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
    weights: Any,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises.

    Also requires the generated program to be registered for the device's
    arch (``cake_backend.generated_program_available``).
    """
    import torch

    try:
        if not (
            flashinfer_module_available(FI_MODULE, FI_BACKEND_MODULE, FI_JIT_MODULE)
            and cuda_tensor_on(pixel_values, ARCHS)
            and pixel_values.dtype == torch.bfloat16
            and pixel_values.is_contiguous()
            and pixel_values.ndim == 4
            and tuple(pixel_values.shape[1:]) == (IN_CH, PATCH, PATCH)
            and pixel_values.data_ptr() % 16 == 0
        ):
            return False
        total = _grid_thws_ok(grid_thws)
        if total is None or total != pixel_values.shape[0]:
            return False
        if not _weights_ok(weights, pixel_values.device):
            return False
        if out is not None:
            merged = sum(int(h) // 2 * (int(w) // 2) for _t, h, w in grid_thws)
            if not (
                out.dtype == torch.bfloat16
                and out.is_contiguous()
                and tuple(out.shape) == (merged, TEXT_HIDDEN)
                and out.device == pixel_values.device
            ):
                return False
        from flashinfer.experimental.kimi_k3_vision_tower.cake_backend import (
            generated_program_available,
        )

        return bool(generated_program_available(pixel_values.device))
    except Exception:
        return False


def kimi_k3_vision_tower(
    pixel_values: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
    weights: Any,
    out: Optional[torch.Tensor] = None,
    *,
    plan: Any = None,
    pos_rows: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward to FlashInfer; returns BF16 ``out [N, 7168]``.

    Dict weights are prepared on every call; pass ``PreparedWeights`` from
    :func:`prepare_kimi_k3_vision_weights` in production.
    """
    from flashinfer.kimi_k3_vision import kimi_k3_vision_tower

    return kimi_k3_vision_tower(
        pixel_values,
        grid_thws,
        weights,
        out,
        plan=plan,
        pos_rows=pos_rows,
        backend="cake",
    )


def prepare_kimi_k3_vision_tower(
    pixel_values: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
    weights: Any,
    out: Optional[torch.Tensor] = None,
    *,
    plan: Any = None,
    pos_rows: Optional[torch.Tensor] = None,
) -> Any:
    """Forward to FlashInfer; returns ``KimiK3VisionTowerRunner``.

    ``runner.launch()`` (alias ``runner()``) writes ``runner.out`` with no
    allocation or host sync; CUDA-graph capturable. Admission:
    :func:`supports_kimi_k3_vision_tower`.
    """
    from flashinfer.kimi_k3_vision import prepare_kimi_k3_vision_tower

    return prepare_kimi_k3_vision_tower(
        pixel_values,
        grid_thws,
        weights,
        out,
        plan=plan,
        pos_rows=pos_rows,
        backend="cake",
    )


def supports_prepare_kimi_k3_vision_weights(weights: Any) -> bool:
    """Admission for the weight preparation (dict of BF16 CUDA tensors); never raises."""
    try:
        if not (
            flashinfer_module_available(FI_BACKEND_MODULE)
            and isinstance(weights, dict)
            and "patch_proj" in weights
        ):
            return False
        device = weights["patch_proj"].device
        return device.type == "cuda" and _weights_ok(weights, device)
    except Exception:
        return False


def prepare_kimi_k3_vision_weights(weights: dict) -> Any:
    """Forward to FlashInfer; returns ``PreparedWeights`` (once per model)."""
    from flashinfer.experimental.kimi_k3_vision_tower.cake_backend import (
        prepare_kimi_k3_vision_weights,
    )

    return prepare_kimi_k3_vision_weights(weights)
