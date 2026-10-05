"""Cake ``FP8_PB_WO`` projection GEMMs for the Kimi-K3 MLA linears.

Opt-in through ``SGLANG_CAKE_ROUTES=kimi_k3_fp8_projection``. Follows the
``_k3_bf16_gemm`` / ``cutedsl_bf16_gemm`` precedent in ``kimi_k3.py``: a
shape-gated GEMM swap that leaves the linear's weights, loading, TP reduction
and output-gate wrappers untouched. The swap is installed at the quant-method
level of each admitted linear (``q_a`` / ``kv_a`` (fused or split), ``q_b``,
``kv_b``, ``o_proj``), so every ``forward`` variant of the linear classes keeps
its collective logic and only the ``apply`` / ``apply_into`` GEMM changes.

Lifecycle (FlashInfer ``gemm.kimi_k3_fp8_projection`` at ``e4f94f948``):

* weights: ``prepare_kimi_k3_fp8_projection_weights`` once per linear, after
  ``process_weights_after_loading`` (the wrapper hooks that call; loaders that
  processed the weights before ``post_load_weights`` are covered by a lazy
  first-call prepare). FlashInfer requantizes to UE8M0 and stores a 256-row
  padded, 128x128-tiled E4M3 copy: roughly one extra copy of the FP8 weight
  per admitted linear stays resident next to the original (the fallback path
  and the other quant-method consumers keep using the original).
* launcher: ``kimi_k3_fp8_projection_launcher(prepared)`` once per linear
  (round 7, CAKE-949). ``launcher(x, out)`` resolves the route plan once per
  ``(M, output row stride, output address class)``, allocates the workspace
  (``[M, K]`` E4M3 + a small byte buffer) once per ``M`` and reuses it (least
  recently used ``M`` evicted beyond 64 cached rows; an ``M`` first launched
  under CUDA-graph capture stays pinned, so a captured graph never addresses
  freed memory), and binds only the call's tensors before the launches. The
  per-call host path of round 6 (workspace allocation + runner preparation +
  launch, ~100 us per call at M = 1024) is what made the route a net loss at
  batch-1 prefill although its kernels were faster. The JIT modules of a route
  are loaded on the first eager call for that ``M``; during capture a
  not-yet-warmed ``M`` falls back to the regular FP8 linear (logged once).
  SGLang's graph runner warms every captured batch size eagerly first.
* legacy runner path (FlashInfer without the launcher entry):
  ``allocate_kimi_k3_fp8_projection_workspace(prepared, M)`` +
  ``prepare_kimi_k3_fp8_projection(x, prepared, out, workspace)`` + ``launch()``
  per call.

Admission (``supports_kimi_k3_fp8_projection_weights`` / ``_projection``):
SM100a / SM103a, E4M3 ``[N, K]`` weight with ``K % 128 == 0`` and the ModelOpt
fp32 block scale ``[ceil(N/128), K/128]``, even ``N``; ``N`` not a multiple of
128 is padded once with zero rows into a transient copy before preparation
(``n_valid = N``). Activations BF16 ``[M, K]`` contiguous, no bias, output BF16
``[M, N]`` (``apply_into`` targets need unit column stride, an even row stride
``>= N`` and 4-byte alignment). Anything else runs the wrapped quant method.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from sglang.kernels.cake_kernels._routes import cake_route_enabled

logger = logging.getLogger(__name__)

ROUTE = "kimi_k3_fp8_projection"
BLOCK = 128
PROJECTIONS = (
    "fused_qkv_a_proj_with_mqa",
    "q_a_proj",
    "kv_a_proj_with_mqa",
    "q_b_proj",
    "kv_b_proj",
    "o_proj",
)
_CAKE_ERRORS = (RuntimeError, ValueError, NotImplementedError)


# Thin indirections to the Cake adapter / facade so the heavy modules load
# lazily and unit tests can substitute them.
def _supports_weights(weight, weight_scale, n_valid) -> bool:
    from sglang.kernels.cake_kernels import gemm_kimi_k3_fp8_projection as cake

    return cake.supports_kimi_k3_fp8_projection_weights(weight, weight_scale, n_valid)


def _supports_projection(x, prepared, out=None, workspace=None) -> bool:
    from sglang.kernels.cake_kernels import gemm_kimi_k3_fp8_projection as cake

    return cake.supports_kimi_k3_fp8_projection(x, prepared, out, workspace)


def _prepare_weights(weight, weight_scale, n_valid):
    from sglang.kernels.ops.gemm.cake import (
        cake_prepare_kimi_k3_fp8_projection_weights,
    )

    return cake_prepare_kimi_k3_fp8_projection_weights(weight, weight_scale, n_valid)


def _allocate_workspace(prepared, m: int):
    from sglang.kernels.ops.gemm.cake import (
        cake_allocate_kimi_k3_fp8_projection_workspace,
    )

    return cake_allocate_kimi_k3_fp8_projection_workspace(prepared, m)


def _prepare_projection(x, prepared, out, workspace):
    from sglang.kernels.ops.gemm.cake import cake_prepare_kimi_k3_fp8_projection

    return cake_prepare_kimi_k3_fp8_projection(x, prepared, out, workspace)


def _make_launcher(prepared) -> Optional[Any]:
    """The per-weight cached launcher, or ``None`` with a FlashInfer that predates it."""
    from sglang.kernels.cake_kernels import gemm_kimi_k3_fp8_projection as cake

    if not cake.supports_kimi_k3_fp8_projection_launcher():
        return None
    from sglang.kernels.ops.gemm.cake import cake_kimi_k3_fp8_projection_launcher

    try:
        return cake_kimi_k3_fp8_projection_launcher(prepared)
    except _CAKE_ERRORS + (AttributeError, TypeError) as exc:
        logger.warning("Cake kimi_k3_fp8_projection: launcher unavailable (%s); using the per-call runner path", exc)
        return None


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def prepare_linear_weight(linear: torch.nn.Module, name: str) -> Optional[Any]:
    """Prepare one linear's serialized FP8_PB_WO weight for Cake; ``None`` when
    the weight is not in the contract (logged once per linear)."""
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scale_inv", None)
    if weight is None or scale is None:
        logger.info("Cake %s: no FP8 block weight / weight_scale_inv; skipped", name)
        return None
    weight = weight.data
    scale = scale.data
    if (
        weight.dtype != torch.float8_e4m3fn
        or weight.ndim != 2
        or not weight.is_contiguous()
        or scale.dtype != torch.float32
    ):
        logger.info(
            "Cake %s: weight %s/%s scale %s is not a serialized FP8_PB_WO block "
            "weight; skipped",
            name,
            tuple(weight.shape),
            weight.dtype,
            scale.dtype,
        )
        return None
    n, k = (int(v) for v in weight.shape)
    n_pad = -(-n // BLOCK) * BLOCK
    scale_2d = scale.reshape(-1, scale.shape[-1]) if scale.ndim == 4 else scale
    if (
        k % BLOCK
        or n % 2
        or scale_2d.ndim != 2
        or tuple(scale_2d.shape) != (n_pad // BLOCK, k // BLOCK)
    ):
        logger.info(
            "Cake %s: shape N=%d K=%d scale %s outside the contract; skipped",
            name,
            n,
            k,
            tuple(scale.shape),
        )
        return None
    scale_2d = scale_2d.contiguous()
    if n == n_pad:
        padded = weight
    else:
        # Serialized checkpoints carry the 128-row block scale but not the
        # padded rows; pad once into a transient copy (freed after prepare).
        padded = torch.zeros((n_pad, k), dtype=weight.dtype, device=weight.device)
        padded[:n].copy_(weight)
    if not _supports_weights(padded, scale_2d, n):
        logger.info(
            "Cake %s: adapter admission rejected N=%d (pad %d) K=%d on %s; skipped",
            name,
            n,
            n_pad,
            k,
            weight.device,
        )
        return None
    prepared = _prepare_weights(padded, scale_2d, n)
    logger.info(
        "Cake %s: prepared FP8 projection N=%d (pad %d) K=%d", name, n, n_pad, k
    )
    return prepared


class CakeFp8ProjectionLinearMethod:
    """Instance-level quant-method wrapper: Cake projection when admitted,
    otherwise the wrapped FP8 method. Every other attribute (``quant_config``,
    ``process_weights_after_loading``, ``weight_block_size``, ...) is delegated.
    """

    def __init__(self, inner: Any, name: str):
        self._inner = inner
        self.name = name
        self.prepared: Optional[Any] = None
        self.n_valid = 0
        self.k = 0
        self._prepare_attempted = False
        self._admitted: dict = {}  # M -> adapter admission for that row count
        self._warm: set = set()  # M values launched eagerly (JIT modules loaded)
        self._launcher: Optional[Any] = None  # per-weight cached launcher (round 7)
        self._logged: set = set()
        if getattr(inner, "apply_into", None) is not None:
            # Only advertise apply_into when the wrapped method has it
            # (RowParallelLinear probes it with getattr).
            self.apply_into = self._apply_into

    def __getattr__(self, attr: str):
        if attr == "_inner":
            raise AttributeError(attr)
        return getattr(self._inner, attr)

    # ---- lifecycle ----

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._inner.process_weights_after_loading(layer)
        self._prepare(layer)

    def _prepare(self, layer: torch.nn.Module) -> None:
        self._prepare_attempted = True
        self.prepared = None
        self._launcher = None
        self._admitted.clear()
        self._warm.clear()
        prepared = prepare_linear_weight(layer, self.name)
        if prepared is None:
            return
        self.prepared = prepared
        self.n_valid = int(prepared.n_valid)
        self.k = int(prepared.K)
        self._launcher = _make_launcher(prepared)
        if self._launcher is None:
            self._log_once(
                "no-launcher",
                "FlashInfer has no kimi_k3_fp8_projection_launcher; using the per-call runner path",
            )

    # ---- GEMM ----

    def apply(self, layer: torch.nn.Module, x, bias: Optional[torch.Tensor] = None):
        out = self._cake(layer, x, None, bias)
        if out is not None:
            return out
        return self._inner.apply(layer, x, bias)

    def _apply_into(
        self,
        layer: torch.nn.Module,
        x,
        out: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        res = self._cake(layer, x, out, bias)
        if res is not None:
            return res
        return self._inner.apply_into(layer, x, out, bias=bias)

    def _log_once(self, key: str, msg: str) -> None:
        if key in self._logged:
            return
        self._logged.add(key)
        logger.info("Cake %s: %s", self.name, msg)

    def _out_ok(self, out: torch.Tensor, m: int, device: torch.device) -> bool:
        return (
            isinstance(out, torch.Tensor)
            and out.ndim == 2
            and out.dtype == torch.bfloat16
            and out.device == device
            and tuple(out.shape) == (m, self.n_valid)
            and out.stride(1) == 1
            and out.stride(0) % 2 == 0
            and out.stride(0) >= self.n_valid
            and (out.storage_offset() * out.element_size()) % 4 == 0
        )

    def _cake(
        self,
        layer: torch.nn.Module,
        x,
        out: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if bias is not None or not isinstance(x, torch.Tensor):
            return None
        capturing = _is_capturing()
        if self.prepared is None:
            if self._prepare_attempted or capturing:
                if capturing and not self._prepare_attempted:
                    self._log_once(
                        "capture-unprepared",
                        "first call happened inside CUDA-graph capture; "
                        "weights were not prepared, using the FP8 linear",
                    )
                return None
            # Loader ran process_weights_after_loading before post_load_weights
            # installed this wrapper: prepare lazily on the first eager call.
            self._prepare(layer)
            if self.prepared is None:
                return None
        if (
            x.ndim != 2
            or x.dtype != torch.bfloat16
            or int(x.shape[1]) != self.k
            or not x.is_contiguous()
        ):
            return None
        m = int(x.shape[0])
        if out is not None and not self._out_ok(out, m, x.device):
            return None
        if capturing and m not in self._warm:
            self._log_once(
                f"capture-{m}",
                f"M={m} reached CUDA-graph capture before an eager warm-up "
                "launch; using the FP8 linear for this graph",
            )
            return None
        admitted = self._admitted.get(m)
        if admitted is None:
            admitted = bool(_supports_projection(x, self.prepared))
            self._admitted[m] = admitted
            if not admitted:
                self._log_once(
                    f"reject-{m}",
                    f"adapter admission rejected M={m} N={self.n_valid} K={self.k}",
                )
        if not admitted:
            return None
        if out is None:
            out = torch.empty((m, self.n_valid), dtype=torch.bfloat16, device=x.device)
        try:
            if self._launcher is not None:
                self._launcher(x, out)
            else:
                workspace = _allocate_workspace(self.prepared, m)
                runner = _prepare_projection(x, self.prepared, out, workspace)
                runner.launch()
        except _CAKE_ERRORS as exc:
            # FlashInfer validates on the host before launching; keep this M
            # on the FP8 linear from now on.
            self._admitted[m] = False
            logger.warning(
                "Cake %s: FlashInfer rejected M=%d N=%d K=%d, using the FP8 "
                "linear for this shape: %s",
                self.name,
                m,
                self.n_valid,
                self.k,
                exc,
            )
            return None
        if not capturing:
            self._warm.add(m)
        return out


def install_cake_kimi_k3_fp8_projections(
    attn: torch.nn.Module,
    is_fp8_pb_wo: Callable[[str], bool],
    prefix: str = "",
) -> list:
    """Wrap the admitted MLA projection linears of ``attn`` for the Cake route.

    ``is_fp8_pb_wo(prefix)`` is the model's ``_uses_modelopt_fp8_pb_wo``
    predicate bound to its quant config; only linears it admits are wrapped.
    Returns the names of the wrapped linears (empty when the route is off).
    The wrapper prepares the Cake weight copy when the loader calls
    ``process_weights_after_loading`` on it, or lazily on the first eager call.
    """
    if not cake_route_enabled(ROUTE):
        return []
    installed = []
    for name in PROJECTIONS:
        linear = getattr(attn, name, None)
        if linear is None or not isinstance(linear, torch.nn.Module):
            continue
        quant_method = getattr(linear, "quant_method", None)
        if quant_method is None or isinstance(
            quant_method, CakeFp8ProjectionLinearMethod
        ):
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        if not is_fp8_pb_wo(full_name):
            logger.info("Cake %s: quant algo is not FP8_PB_WO; skipped", full_name)
            continue
        linear.quant_method = CakeFp8ProjectionLinearMethod(quant_method, full_name)
        installed.append(name)
    return installed


__all__ = (
    "ROUTE",
    "PROJECTIONS",
    "CakeFp8ProjectionLinearMethod",
    "install_cake_kimi_k3_fp8_projections",
    "prepare_linear_weight",
)
