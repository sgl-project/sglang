from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.cake_kernels._routes import cake_route_enabled
from sglang.kernels.ops.moe.dsv4 import (
    silu_and_mul_clamp,
    silu_and_mul_masked_post_quant,
)
from sglang.kernels.ops.moe.triton_pad_expert_counts import pad_expert_counts
from sglang.kernels.ops.quantization.per_token_group_quant import per_token_group_quant

logger = logging.getLogger(__name__)

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner import deep_gemm_sm120
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_a2a_backend
from sglang.srt.runtime_context import (
    get_exec,
    get_flags,
    get_parallel,
)
from sglang.srt.utils import (
    ceil_div,
    dispose_tensor,
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
)
from sglang.srt.utils.offloader import get_offloader

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
        DeepEPv2CombineInput,
        DeepEPv2DispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_cuda = is_cuda()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_musa = is_musa()


if not (_is_npu or _is_hip) and _is_cuda:
    from sglang.kernels.ops.activation.activation import (
        silu_and_mul as _legacy_silu_and_mul,
    )
elif _is_musa:
    _silu_and_mul_musa = torch.nn.SwishGLU()
else:
    _legacy_silu_and_mul = None


_DEEPGEMM_ON_H20 = get_bool_env_var("SGLANG_DEEPGEMM_ON_H20")
_masked_standard_layout_memory_budget_bytes: Optional[int] = None


# ---------------------------------------------------------------------------
# Cake (FlashInfer) contiguous grouped FP8 GEMM route, opt-in via
# ``SGLANG_CAKE_ROUTES=moe_fp8_grouped``.
#
# The FlashInfer prepared runners bind tensor *storage* (addresses, shapes,
# dtypes) at preparation and their first ``launch()`` is not CUDA-graph
# capturable.  The DeepGEMM runner input tensors are freshly allocated by the
# dispatcher on every call, so this route owns one static buffer arena per
# (device, K, N), sized to the largest M seen and shared by every MoE layer and
# every M (runners are prepared on leading-row views of it), copies the
# dispatcher tensors into it, and keeps one prepared runner pair per
# (arena, M, expert weights).
# Admission is all-or-nothing: both expert GEMMs run on Cake or the call falls
# through to the unchanged DeepGEMM code below.
# ---------------------------------------------------------------------------

_CAKE_ROUTE = "moe_fp8_grouped"
_CAKE_SCALE_BLOCK = 128
_CAKE_SILU_MAX_M = 8192
_CAKE_SILU_K_MULTIPLE = 512
_CAKE_SILU_N_MULTIPLE = 256
_cake_logged: Dict[str, bool] = {}


@lru_cache(maxsize=1)
def _cake_grouped_fp8_api() -> SimpleNamespace:
    """Lazy handles to the Cake adapter admission checks and prepared-runner wrappers."""
    from sglang.kernels.cake_kernels import gemm_grouped_fp8 as adapter
    from sglang.kernels.ops.gemm.cake import (
        cake_prepare_group_gemm_fp8_nt_groupwise_contiguous,
        cake_prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant,
    )

    return SimpleNamespace(
        supports_plain=adapter.supports_group_gemm_fp8_nt_groupwise_contiguous,
        supports_fused=adapter.supports_group_gemm_fp8_nt_groupwise_contiguous_silu_quant,
        prepare_plain=cake_prepare_group_gemm_fp8_nt_groupwise_contiguous,
        prepare_fused=cake_prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant,
    )


def _cake_stream_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _cake_log_once(key: str, message: str) -> None:
    if not _cake_logged.get(key):
        _cake_logged[key] = True
        logger.info(message)


def _cake_unpack_ue8m0_into(dst: torch.Tensor, packed: torch.Tensor) -> None:
    """Write ``2 ** (byte - 127)`` of each packed UE8M0 byte into FP32 ``dst``.

    ``packed`` is the DeepGEMM int32 layout: one int32 holds four consecutive
    K-group exponents (byte 0 = lowest group); any strides are accepted.
    ``dst`` is contiguous FP32 ``(*, k_groups)`` with ``k_groups <= 4 * packed.shape[-1]``.
    """
    rows = packed.shape[:-1]
    k_groups = dst.shape[-1]
    bytes_ = packed.contiguous().view(torch.uint8).view(*rows, 4 * packed.shape[-1])
    torch.bitwise_left_shift(
        bytes_[..., :k_groups].to(torch.int32), 23, out=dst.view(torch.int32)
    )


def _cake_fill_activation_scale(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy the dispatcher's activation scales into the FP32 ``(M, K/128)`` buffer."""
    if src.dtype == torch.float32:
        dst.copy_(src)
    else:
        _cake_unpack_ue8m0_into(dst, src)


def _cake_activation_scale_ok(src: torch.Tensor, m: int, k_groups: int) -> bool:
    if src.ndim != 2 or int(src.shape[0]) != m:
        return False
    if src.dtype == torch.float32:
        return int(src.shape[1]) == k_groups
    return src.dtype == torch.int32 and 4 * int(src.shape[1]) >= k_groups


def _cake_debug_sync(stage: str) -> None:
    """``SGLANG_CAKE_DEBUG``: synchronize after ``stage`` (outside graph capture) and log the outcome."""
    if torch.cuda.is_current_stream_capturing():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - diagnostics must name the failing stage
        logger.error("Cake %s debug: stage %s FAILED: %r", _CAKE_ROUTE, stage, exc)
        raise
    logger.info("Cake %s debug: stage %s ok", _CAKE_ROUTE, stage)


def _cake_debug_tensor(name: str, t: Optional[torch.Tensor]) -> str:
    if t is None:
        return f"{name}=None"
    return (
        f"{name}=shape{tuple(t.shape)}/{str(t.dtype).replace('torch.', '')}"
        f"/stride{tuple(t.stride())}/ptr%16={t.data_ptr() % 16}"
    )


def _cake_debug_describe(req: _CakeContigRequest, plan: _CakeContigPlan) -> None:
    """``SGLANG_CAKE_DEBUG``: one-time summary of the dispatcher inputs and the staged Cake buffers."""
    b = plan.buffers
    mi = b["m_indices"]
    raw = req.m_indices
    parts = [
        _cake_debug_tensor("hidden_states", req.hidden_states),
        _cake_debug_tensor("hidden_states_scale", req.hidden_states_scale),
        _cake_debug_tensor("m_indices", raw),
        _cake_debug_tensor("w13_weight", req.w13_weight),
        _cake_debug_tensor("w13_scale", req.w13_scale),
        _cake_debug_tensor("w2_weight", req.w2_weight),
        _cake_debug_tensor("w2_scale", req.w2_scale),
        f"layout_alignment={req.layout_alignment} swiglu_limit={req.swiglu_limit}",
        f"m_indices_raw[min={int(raw.min())} max={int(raw.max())} neg={int((raw < 0).sum())}]",
        f"m_indices_staged[min={int(mi.min())} max={int(mi.max())} "
        f"nondecreasing={bool((mi[1:] >= mi[:-1]).all())} groups={int(req.w13_weight.shape[0])}]",
    ]
    for name in ("a_scale",):
        t = b[name].float()
        parts.append(
            f"{name}[finite={bool(torch.isfinite(t).all())} min={t.min().item():.3g} max={t.max().item():.3g}]"
        )
    for name, runner in (("gateup", plan.gateup_runner), ("down", plan.down_runner)):
        parts.append(
            f"{name}_runner[route={getattr(runner, 'route', None)} grid={getattr(runner, 'grid', None)} "
            f"module={getattr(runner, 'module_name', None)}]"
        )
    parts.append(" ".join(_cake_debug_tensor(k, v) for k, v in b.items()))
    logger.info("Cake %s debug: fused=%s %s", _CAKE_ROUTE, plan.fused, " ".join(parts))


_cake_weight_scale_cache: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor] = {}


def _cake_weight_scale_fp32(
    scale: torch.Tensor, groups: int, n: int, k: int
) -> Optional[torch.Tensor]:
    """Block scales as contiguous FP32 ``(G, N/128, K/128)`` or ``None`` if unrecognized.

    Accepts the plain FP32 block layout and the DeepGEMM packed UE8M0 int32
    layouts (``(G, N, ceil(K/512))`` row-repeated as produced by
    ``transform_scale_ue8m0`` or ``(G, N/128, ceil(K/512))``).  Converted
    tensors are cached per storage; weights are static after loading.
    """
    n_groups, k_groups = n // _CAKE_SCALE_BLOCK, k // _CAKE_SCALE_BLOCK
    target = (groups, n_groups, k_groups)
    if scale.dtype == torch.float32:
        if tuple(scale.shape) != target:
            return None
        return scale if scale.is_contiguous() else scale.contiguous()
    if scale.dtype != torch.int32 or scale.ndim != 3 or int(scale.shape[0]) != groups:
        return None
    key = (scale.data_ptr(), tuple(scale.shape))
    cached = _cake_weight_scale_cache.get(key)
    if cached is not None:
        return cached
    rows = int(scale.shape[1])
    if rows == n:
        packed = scale[:, ::_CAKE_SCALE_BLOCK, :]
    elif rows == n_groups:
        packed = scale
    else:
        return None
    if 4 * int(packed.shape[-1]) < k_groups:
        return None
    out = torch.empty(target, dtype=torch.float32, device=scale.device)
    _cake_unpack_ue8m0_into(out, packed)
    _cake_weight_scale_cache[key] = out
    return out


@dataclass
class _CakeContigRequest:
    """Everything the Cake route needs; building it runs no kernels."""

    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    m_indices: torch.Tensor
    w13_weight: torch.Tensor
    w13_scale: Optional[torch.Tensor]
    w2_weight: torch.Tensor
    w2_scale: Optional[torch.Tensor]
    activation: str
    swiglu_limit: Optional[float]
    silu_mul_keep_fp32: bool
    use_swizzle: bool
    use_mxfp8: bool
    is_fp4_experts: bool
    activation_scale_block_size: Optional[int]
    # Row alignment of expert boundaries in the contiguous layout (None = 128).
    layout_alignment: Optional[int]


@dataclass
class _CakeContigPlan:
    fused: bool
    buffers: Dict[str, torch.Tensor]
    gateup_runner: Any
    down_runner: Any
    swiglu_limit: Optional[float]
    debug_described: bool = False


@dataclass
class _CakeArena:
    """Full-capacity static buffers for one ``(device, K, N)``; plans use ``[:M]`` views."""

    serial: int
    rows: int
    buffers: Dict[str, torch.Tensor]


class _CakeContigFp8Route:
    """Static buffer arena + prepared Cake runners for the contiguous FP8 expert GEMMs.

    FlashInfer binds ``M`` (the padded row count ``all_tokens``) at preparation,
    so runners are cached per exact ``M``; their tensors are leading-row views
    of one arena per ``(device, K, N)`` sized to the largest ``M`` seen, which
    costs ``M_max * (K + K/32 + 4 + N/2 + N/64 + 2K)`` bytes (plus
    ``2 * M_max * N`` when the fused SwiGLU route is not admitted and a BF16
    gate_up buffer is needed); e.g. 6.7 KiB per row for K=2048, N=1024, i.e.
    one ~1.1 GiB arena for a 16384-token prefill instead of one set per captured
    batch size.  An arena that must grow is replaced (its plans are dropped and
    rebuilt) and the old one is retained because CUDA graphs captured against
    it still replay into it.  Prepared runners are per layer (weight storage)
    and hold only descriptor workspace.
    """

    def __init__(self) -> None:
        self._arenas: Dict[Tuple[Any, ...], _CakeArena] = {}
        self._retired_arenas: List[_CakeArena] = []
        self._arena_serial = 0
        self._plans: Dict[Tuple[Any, ...], Optional[_CakeContigPlan]] = {}

    # -- admission ---------------------------------------------------------

    @staticmethod
    def _static_reject_reason(req: _CakeContigRequest) -> Optional[str]:
        if req.activation != "silu":
            return f"activation {req.activation!r} (only silu)"
        if req.use_mxfp8 or req.is_fp4_experts:
            return "mxfp8 / fp4 expert weights"
        if req.use_swizzle:
            return "swizzled contiguous layout"
        if req.activation_scale_block_size not in (None, _CAKE_SCALE_BLOCK):
            return f"activation scale block {req.activation_scale_block_size}"
        hs, w13, w2 = req.hidden_states, req.w13_weight, req.w2_weight
        if (
            hs.dtype != torch.float8_e4m3fn
            or w13.dtype != torch.float8_e4m3fn
            or w2.dtype != torch.float8_e4m3fn
        ):
            return "non-E4M3 operands"
        if req.w13_scale is None or req.w2_scale is None:
            return "missing weight scales"
        if hs.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
            return "unexpected operand ranks"
        m, k = (int(v) for v in hs.shape)
        groups, n, k13 = (int(v) for v in w13.shape)
        groups2, k2, h2 = (int(v) for v in w2.shape)
        if m <= 0:
            return "empty batch"
        if k13 != k or groups2 != groups or k2 != k or 2 * h2 != n:
            return f"inconsistent expert shapes w13={tuple(w13.shape)} w2={tuple(w2.shape)}"
        if k % _CAKE_SCALE_BLOCK or n % (2 * _CAKE_SCALE_BLOCK):
            return f"K={k} must be a multiple of 128 and N={n} of 256"
        if req.m_indices.dtype != torch.int32 or tuple(req.m_indices.shape) != (m,):
            return "m_indices must be int32 (M,)"
        if not _cake_activation_scale_ok(
            req.hidden_states_scale, m, k // _CAKE_SCALE_BLOCK
        ):
            return (
                "unrecognized activation scale layout "
                f"{tuple(req.hidden_states_scale.shape)} {req.hidden_states_scale.dtype}"
            )
        return None

    @staticmethod
    def _fused_eligible(req: _CakeContigRequest, m: int, n: int, k: int) -> bool:
        alignment = req.layout_alignment or _CAKE_SCALE_BLOCK
        return (
            m <= _CAKE_SILU_MAX_M
            and k % _CAKE_SILU_K_MULTIPLE == 0
            and n % _CAKE_SILU_N_MULTIPLE == 0
            and alignment % _CAKE_SCALE_BLOCK == 0
            and req.swiglu_limit is None
            and not req.silu_mul_keep_fp32
        )

    def _arena(
        self, device: torch.device, m: int, k: int, n: int, *, grow: bool
    ) -> Optional[_CakeArena]:
        """The arena for ``(device, K, N)`` with at least ``m`` rows; ``None`` if it
        would have to be (re)allocated and ``grow`` is false."""
        key = (device.type, device.index, k, n)
        arena = self._arenas.get(key)
        if arena is not None and arena.rows >= m:
            return arena
        if not grow:
            return None
        rows = m
        if arena is not None:
            # Captured graphs replay into the old buffers: keep them alive.
            self._retired_arenas.append(arena)
            rows = max(m, arena.rows)
        h = n // 2
        self._arena_serial += 1
        # One contiguous allocation (one allocator carve-out instead of seven)
        # viewed as the individual buffers; every view starts 256-byte aligned.
        specs = [
            ("a", (rows, k), torch.float8_e4m3fn),
            ("a_scale", (rows, k // _CAKE_SCALE_BLOCK), torch.float32),
            ("m_indices", (rows,), torch.int32),
            ("act", (rows, h), torch.float8_e4m3fn),
            ("act_scale", (rows, h // _CAKE_SCALE_BLOCK), torch.float32),
            ("down_out", (rows, k), torch.bfloat16),
            ("gateup", (rows, n), torch.bfloat16),
        ]
        offsets, total = [], 0
        for _, shape, dtype in specs:
            offsets.append(total)
            total += (
                -(
                    -math.prod(shape)
                    * torch.tensor([], dtype=dtype).element_size()
                    // 256
                )
                * 256
            )
        storage = torch.empty((total,), dtype=torch.uint8, device=device)
        buffers = {
            name: storage[
                off : off
                + math.prod(shape) * torch.tensor([], dtype=dtype).element_size()
            ]
            .view(dtype)
            .view(shape)
            for (name, shape, dtype), off in zip(specs, offsets)
        }
        arena = _CakeArena(serial=self._arena_serial, rows=rows, buffers=buffers)
        arena.buffers["_storage"] = storage
        self._arenas[key] = arena
        return arena

    def _get_buffers(
        self, device: torch.device, m: int, k: int, n: int, fused: bool
    ) -> Dict[str, torch.Tensor]:
        """Leading-row ``[:m]`` views (contiguous, same base alignment) of the arena."""
        arena = self._arena(device, m, k, n, grow=True)
        return {name: t[:m] for name, t in arena.buffers.items() if name != "_storage"}

    def _build_plan(
        self, req: _CakeContigRequest
    ) -> Tuple[Optional[_CakeContigPlan], str]:
        api = _cake_grouped_fp8_api()
        m, k = (int(v) for v in req.hidden_states.shape)
        groups, n, _ = (int(v) for v in req.w13_weight.shape)
        h = n // 2
        w13_scale = _cake_weight_scale_fp32(req.w13_scale, groups, n, k)
        w2_scale = _cake_weight_scale_fp32(req.w2_scale, groups, k, h)
        if w13_scale is None or w2_scale is None:
            return None, (
                "unrecognized weight scale layout "
                f"w13={tuple(req.w13_scale.shape)} {req.w13_scale.dtype} "
                f"w2={tuple(req.w2_scale.shape)} {req.w2_scale.dtype}"
            )
        device = req.hidden_states.device
        fused = self._fused_eligible(req, m, n, k)
        buffers = self._get_buffers(device, m, k, n, fused)
        if fused and not api.supports_fused(
            buffers["a"],
            req.w13_weight,
            buffers["a_scale"],
            w13_scale,
            buffers["m_indices"],
            buffers["act"],
            buffers["act_scale"],
        ):
            fused = False
            buffers = self._get_buffers(device, m, k, n, fused)
        if not fused and not api.supports_plain(
            buffers["a"],
            req.w13_weight,
            buffers["a_scale"],
            w13_scale,
            buffers["m_indices"],
            buffers["gateup"],
        ):
            return None, f"gate_up GEMM not admitted (M={m} N={n} K={k} G={groups})"
        if not api.supports_plain(
            buffers["act"],
            req.w2_weight,
            buffers["act_scale"],
            w2_scale,
            buffers["m_indices"],
            buffers["down_out"],
        ):
            return None, f"down GEMM not admitted (M={m} N={k} K={h} G={groups})"
        if fused:
            gateup_runner = api.prepare_fused(
                buffers["a"],
                req.w13_weight,
                buffers["a_scale"],
                w13_scale,
                buffers["m_indices"],
                buffers["act"],
                buffers["act_scale"],
            )
        else:
            gateup_runner = api.prepare_plain(
                buffers["a"],
                req.w13_weight,
                buffers["a_scale"],
                w13_scale,
                buffers["m_indices"],
                buffers["gateup"],
            )
        down_runner = api.prepare_plain(
            buffers["act"],
            req.w2_weight,
            buffers["act_scale"],
            w2_scale,
            buffers["m_indices"],
            buffers["down_out"],
        )
        if device.type == "cuda":
            # The prepared runners' first launch() initializes their private TMA
            # descriptor storage with a synchronous host-to-device copy.  That
            # storage is a caching-allocator block; a kernel still queued on this
            # stream that writes the block's previous tenant would overwrite the
            # descriptors behind the copy.  Drain the stream once per plan so
            # every kernel queued before the storage was allocated has finished.
            torch.cuda.current_stream(device).synchronize()
        plan = _CakeContigPlan(
            fused=fused,
            buffers=buffers,
            gateup_runner=gateup_runner,
            down_runner=down_runner,
            swiglu_limit=req.swiglu_limit,
        )
        return plan, f"fused_silu_quant={fused} M={m} N={n} K={k} G={groups}"

    # -- execution ---------------------------------------------------------

    @staticmethod
    def _plan_key(req: _CakeContigRequest) -> Tuple[Any, ...]:
        hs = req.hidden_states
        return (
            hs.device.type,
            hs.device.index,
            tuple(int(v) for v in hs.shape),
            tuple(int(v) for v in req.w13_weight.shape),
            req.w13_weight.data_ptr(),
            req.w13_scale.data_ptr() if req.w13_scale is not None else None,
            req.w2_weight.data_ptr(),
            req.w2_scale.data_ptr() if req.w2_scale is not None else None,
            req.swiglu_limit,
            req.silu_mul_keep_fp32,
            req.layout_alignment,
        )

    def try_run(
        self,
        req: _CakeContigRequest,
        allocate_output: Callable[[], torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Run both expert GEMMs on Cake, or return ``None`` to use DeepGEMM.

        Returning ``None`` leaves every request tensor untouched.  ``allocate_output``
        provides the ``(M, K)`` BF16 result tensor (caller-owned allocation), which
        receives a copy of the static Cake output so the returned tensor has the
        same ownership as the DeepGEMM path.
        """
        reason = self._static_reject_reason(req)
        if reason is not None:
            _cake_log_once(
                "fallback", f"Cake {_CAKE_ROUTE}: DeepGEMM fallback, {reason}"
            )
            return None
        hs = req.hidden_states
        m, k = (int(v) for v in hs.shape)
        n = int(req.w13_weight.shape[1])
        capturing = _cake_stream_capturing()
        arena = self._arena(hs.device, m, k, n, grow=not capturing)
        if arena is None:
            _cake_log_once(
                "capture_fallback",
                f"Cake {_CAKE_ROUTE}: shape {tuple(hs.shape)} first seen inside "
                "CUDA-graph capture; using DeepGEMM for this graph "
                "(warm the shape up eagerly before capture)",
            )
            return None
        key = (arena.serial,) + self._plan_key(req)
        if key in self._plans:
            plan = self._plans[key]
            if plan is None:
                return None
        else:
            if capturing:
                _cake_log_once(
                    "capture_fallback",
                    f"Cake {_CAKE_ROUTE}: shape {tuple(hs.shape)} first "
                    "seen inside CUDA-graph capture; using DeepGEMM for this graph "
                    "(warm the shape up eagerly before capture)",
                )
                return None
            plan, detail = self._build_plan(req)
            self._plans[key] = plan
            if plan is None:
                _cake_log_once(
                    "fallback", f"Cake {_CAKE_ROUTE}: DeepGEMM fallback, {detail}"
                )
                return None
            _cake_log_once("route", f"Cake {_CAKE_ROUTE}: route taken, {detail}")
        return self._run(plan, req, allocate_output)

    @staticmethod
    def _run(
        plan: _CakeContigPlan,
        req: _CakeContigRequest,
        allocate_output: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        buffers = plan.buffers
        debug = envs.SGLANG_CAKE_DEBUG.get()
        buffers["a"].copy_(req.hidden_states)
        _cake_fill_activation_scale(buffers["a_scale"], req.hidden_states_scale)
        # ep_scatter marks alignment-padding rows with -1, which FlashInfer
        # rejects; map them onto the preceding expert (their output rows are
        # dropped by post-permute, exactly as DeepGEMM's skipped tiles are).
        filled, _ = torch.cummax(req.m_indices, 0)
        torch.clamp(filled, min=0, out=buffers["m_indices"])
        if debug:
            _cake_debug_sync("inputs_staged")
            if not plan.debug_described:
                plan.debug_described = True
                _cake_debug_describe(req, plan)
        if plan.fused:
            plan.gateup_runner.launch()
            if debug:
                _cake_debug_sync("gateup_fused_silu_quant")
        else:
            from sglang.kernels.ops.moe.dsv4 import silu_and_mul_contig_post_quant

            plan.gateup_runner.launch()
            if debug:
                _cake_debug_sync("gateup")
            silu_and_mul_contig_post_quant(
                input=buffers["gateup"],
                output=buffers["act"],
                output_scale=buffers["act_scale"],
                quant_group_size=_CAKE_SCALE_BLOCK,
                scale_ue8m0=False,
                transposed=False,
                swiglu_limit=plan.swiglu_limit,
                swizzle=False,
            )
            if debug:
                _cake_debug_sync("silu_quant")
        plan.down_runner.launch()
        if debug:
            _cake_debug_sync("down")
        out = allocate_output()
        out.copy_(buffers["down_out"])
        return out

    def reset_for_tests(self) -> None:
        self._arenas.clear()
        self._retired_arenas.clear()
        self._plans.clear()
        _cake_weight_scale_cache.clear()
        _cake_logged.clear()


_CAKE_CONTIG_FP8 = _CakeContigFp8Route()


# TODO(kaixih@nvidia): ideally we should merge this logic into
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.
@torch.compile(disable=_is_hip or _is_npu)
def _cast_to_e8m0_with_rounding_up(x: torch.Tensor) -> torch.Tensor:
    temp = x.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(temp, 23)
    mant = torch.bitwise_and(temp, 0x7FFFFF)
    is_ru = torch.logical_and(
        torch.logical_and((mant > 0), (exp != 0xFE)),
        ~torch.logical_and((exp == 0), (mant <= 0x400000)),
    )
    exp = torch.where(is_ru, exp + 1, exp)
    new_x = exp.to(torch.uint8).view(torch.int)
    return new_x.transpose(1, 2).contiguous().transpose(1, 2)


def copy_list_to_gpu_no_ce(arr: List[int]):
    from sgl_kernel.elementwise import copy_to_gpu_no_ce

    tensor_cpu = torch.tensor(arr, dtype=torch.int32, device="cpu")
    tensor_gpu = torch.empty_like(tensor_cpu, device="cuda")
    copy_to_gpu_no_ce(tensor_cpu, tensor_gpu)
    return tensor_gpu


def set_masked_standard_layout_memory_budget(
    available_memory_bytes: int,
) -> int:
    """Cache the masked-layout share of free non-static device memory."""
    global _masked_standard_layout_memory_budget_bytes
    fraction = envs.SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION.get()
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION must be in (0, 1]"
        )
    _masked_standard_layout_memory_budget_bytes = int(available_memory_bytes * fraction)
    return _masked_standard_layout_memory_budget_bytes


def _estimate_masked_standard_layout_peak_bytes(
    runner_config: MoeRunnerConfig,
    quant_info: DeepGemmMoeQuantInfo,
    hidden_states: torch.Tensor,
) -> int:
    padded_m = (hidden_states.shape[0] // 256 + 1) * 256
    activation_dtype = (
        torch.bfloat16
        if quant_info.w13_weight.dtype == torch.bfloat16
        else torch.float8_e4m3fn
    )
    hidden_size = hidden_states.shape[1]
    gateup_size = quant_info.w13_weight.shape[1]
    gateup_row_bytes = gateup_size * torch.bfloat16.itemsize
    down_output_row_bytes = quant_info.w2_weight.shape[1] * torch.bfloat16.itemsize
    input_row_bytes = hidden_size * activation_dtype.itemsize
    down_input_row_bytes = gateup_size // 2 * activation_dtype.itemsize

    if activation_dtype == torch.bfloat16:
        input_scale_row_bytes = 0
        down_scale_row_bytes = 0
    else:
        block_k = quant_info.block_shape[1] if quant_info.block_shape else 128
        packed_scales = quant_info.use_mxfp8 or deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        scale_item_bytes = (
            torch.uint8.itemsize if packed_scales else torch.float32.itemsize
        )
        input_scale_row_bytes = ceil_div(hidden_size, block_k) * scale_item_bytes
        down_scale_row_bytes = ceil_div(gateup_size // 2, block_k) * scale_item_bytes

    peak_row_bytes = max(
        input_row_bytes + input_scale_row_bytes + gateup_row_bytes,
        gateup_row_bytes + down_input_row_bytes + down_scale_row_bytes,
        down_input_row_bytes + down_scale_row_bytes + down_output_row_bytes,
    )
    return runner_config.num_local_experts * padded_m * peak_row_bytes


_masked_activation_fallback_logged = False


# Masked clamped/swizzled activation only exists as the DSV4 JIT kernel, which
# requires D // 8 >= E and group 128 (silu_and_mul_masked_post_quant.cuh:245).
def _masked_activation_unsupported_reason(
    runner_config: MoeRunnerConfig, quant_info: DeepGemmMoeQuantInfo
) -> Optional[str]:
    if runner_config.swiglu_limit is None and not get_moe_a2a_backend().is_megamoe():
        return None
    d = runner_config.intermediate_size_per_partition
    e = runner_config.num_local_experts
    if d is None or e is None:
        return None
    group_size = quant_info.block_shape[1] if quant_info.block_shape else 128
    if d // 8 < e:
        return f"D // 8 ({d // 8}) < num_local_experts ({e})"
    if group_size != 128:
        return (
            f"masked activation group_size {group_size}, DSV4 JIT kernel requires 128"
        )
    if d % (group_size * 4) != 0:
        return f"D ({d}) not divisible by 4 * group_size"
    return None


def _should_use_masked_standard_layout(
    runner_config: MoeRunnerConfig,
    quant_info: DeepGemmMoeQuantInfo,
    hidden_states: torch.Tensor,
) -> bool:
    # Preserve the Oakhaven WideEP escape hatch while adopting upstream's
    # memory-budget-based auto policy. CUDA graph capture remains masked.
    if (
        envs.SGLANG_OPT_DG_COMPACT_EAGER.get()
        and not get_flags().capture.disable_dispose_tensor
    ):
        return False

    reason = _masked_activation_unsupported_reason(runner_config, quant_info)
    if reason is not None:
        global _masked_activation_fallback_logged
        if not _masked_activation_fallback_logged:
            _masked_activation_fallback_logged = True
            logger.info(
                "DeepGEMM masked standard layout disabled: %s. "
                "Clamped/swizzled activations on this config must use the "
                "compact layout.",
                reason,
            )
        return False
    mode = envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.get().lower()
    if mode not in ("auto", "masked", "compact"):
        raise ValueError(
            "SGLANG_DEEPGEMM_STANDARD_LAYOUT must be one of: auto, masked, compact"
        )
    if mode != "auto":
        return mode == "masked"

    global _masked_standard_layout_memory_budget_bytes
    if _masked_standard_layout_memory_budget_bytes is None:
        # Serving sets an all-rank budget before capture. Direct eager callers
        # fall back to this rank's free memory without querying inside capture.
        # Import lazily to avoid a module-initialization cycle through
        # runner_utils -> DeepEP -> MoE -> this module.
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        if get_is_capture_mode():
            return False
        free_memory, _ = torch.cuda.mem_get_info(hidden_states.device)
        set_masked_standard_layout_memory_budget(free_memory)

    return (
        _estimate_masked_standard_layout_peak_bytes(
            runner_config, quant_info, hidden_states
        )
        <= _masked_standard_layout_memory_budget_bytes
    )


def _get_compact_all_tokens(
    num_assignments: int, num_experts: int, block_e: int = 128
) -> int:
    """Return the maximum padded rows over all routings of the assignments."""
    max_nonempty_experts = min(num_assignments, num_experts)
    return block_e * (
        max_nonempty_experts + (num_assignments - max_nonempty_experts) // block_e
    )


@dataclass
class DeepGemmRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    use_masked_gemm: bool
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None
    m_indices: Optional[torch.Tensor] = None
    # Number of activation elements sharing one scale along K.
    # Records the actual input quantization group, independently of weight scales.
    activation_scale_block_size: Optional[int] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@dataclass
class DeepGemmMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    use_fp8: bool
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None
    # DSV4 mxfp4 layout flag; selects recipe_a=(1,128)/recipe_b=(1,32) downstream.
    is_fp4_experts: bool = False
    use_mxfp8: bool = False

    def __post_init__(self):
        if self.use_mxfp8:
            assert self.block_shape == [
                1,
                32,
            ], f"MXFP8 requires block_shape [1, 32], got {self.block_shape}"
            assert deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0, (
                "MXFP8 requires DEEPGEMM_SCALE_UE8M0=True"
            )

    def scale_recipes(
        self,
        *,
        activation_block_size: Optional[int],
        hidden_size: int,
        activation_scale_width: int,
    ) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
        """Return DeepGEMM A/B scale recipes from explicit layout metadata."""
        if self.use_mxfp8:
            assert self.block_shape is not None
            weight_recipe = (self.block_shape[0], self.block_shape[1])
            activation_block_size = activation_block_size or self.block_shape[1]
            assert ceil_div(hidden_size, activation_block_size * 4) == (
                activation_scale_width
            ), (
                "MXFP8 activation scale mismatch: "
                f"block_size={activation_block_size}, K={hidden_size}, "
                f"scale_width={activation_scale_width}, expected "
                f"{ceil_div(hidden_size, activation_block_size * 4)}"
            )
            return (self.block_shape[0], activation_block_size), weight_recipe
        if self.is_fp4_experts:
            return (1, 128), (1, 32)
        return None, None


class DeepGemmRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        if config.silu_mul_keep_fp32 and (
            config.activation != "silu"
            or not config.is_gated
            or config.gemm1_alpha is not None
        ):
            raise ValueError(
                "silu_mul_keep_fp32 requires gated SiLU without gemm1_alpha"
            )
        # SiTU (Kimi K3) is applied outside the GEMMs in python, so it only
        # needs the masked-gemm activation site to branch (see _run_masked_gemm).
        assert self.config.activation in ("silu", "situ")
        assert self.config.is_gated
        self.swiglu_limit = self.config.swiglu_limit
        # SM120's contiguous GEMM only consumes standard-layout activations, so
        # it opts out of swizzle regardless of the a2a backend.
        self.use_swizzle = (
            get_moe_a2a_backend().is_megamoe() and deep_gemm_sm120.use_swizzle()
        )

    def run(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> DeepGemmRunnerOutput:
        weight_dtype = quant_info.w13_weight.dtype
        if self.config.silu_mul_keep_fp32 and weight_dtype != torch.float8_e4m3fn:
            raise ValueError("silu_mul_keep_fp32 requires FP8 expert weights")
        alignment = (
            running_state.get("contiguous_layout_alignment")
            if not runner_input.use_masked_gemm
            else None
        )
        with deep_gemm_wrapper.contiguous_layout_alignment_scope(alignment):
            if not runner_input.use_masked_gemm:
                if weight_dtype == torch.bfloat16:
                    hidden_states = self._run_bf16_contiguous_gemm(
                        runner_input, quant_info, running_state
                    )
                else:
                    hidden_states = self._run_contiguous_gemm(
                        runner_input, quant_info, running_state
                    )
            else:
                if weight_dtype == torch.bfloat16:
                    hidden_states = self._run_masked_bf16_gemm(
                        runner_input, quant_info, running_state
                    )
                else:
                    hidden_states = self._run_masked_gemm(
                        runner_input, quant_info, running_state
                    )
        return DeepGemmRunnerOutput(hidden_states=hidden_states)

    @staticmethod
    def _allocate_down_output(
        all_tokens: int, hidden_size: int, device: torch.device
    ) -> torch.Tensor:
        # Allocate the MoE output in the NCCL symmetric memory pool when symmetric
        # allocation is required, so the downstream all-reduce takes the low-latency
        # symmetric path. Only this final output enters the pool; intermediate
        # buffers stay on the default allocator to bound pool occupancy.
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            return torch.empty(
                (all_tokens, hidden_size), device=device, dtype=torch.bfloat16
            )

    def _run_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.kernels.ops.moe.dsv4 import silu_and_mul_contig_post_quant
        from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale
        from sglang.kernels.ops.quantization.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
        )

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_dtype = running_state["hidden_states_dtype"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices
        trace_deepep_v2_contig = (
            envs.SGLANG_DEEPEP_V2_TRACE_CONTIG.get()
            and running_state.get("deepep_v2_expanded", False)
        )

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]
        scale_block_size = quant_info.block_shape[1] if quant_info.use_mxfp8 else 128

        if all_tokens == 0:
            if trace_deepep_v2_contig:
                logger.warning("DeepEP v2 expanded contig runner empty return")
            dispose_tensor(hidden_states)
            dispose_tensor(hidden_states_scale)
            return torch.empty(
                (0, K), device=hidden_states_device, dtype=torch.bfloat16
            )

        if cake_route_enabled(_CAKE_ROUTE):
            cake_output = _CAKE_CONTIG_FP8.try_run(
                _CakeContigRequest(
                    hidden_states=hidden_states,
                    hidden_states_scale=hidden_states_scale,
                    m_indices=m_indices,
                    w13_weight=quant_info.w13_weight,
                    w13_scale=quant_info.w13_scale,
                    w2_weight=quant_info.w2_weight,
                    w2_scale=quant_info.w2_scale,
                    activation=self.config.activation,
                    swiglu_limit=self.swiglu_limit,
                    silu_mul_keep_fp32=bool(self.config.silu_mul_keep_fp32),
                    use_swizzle=bool(self.use_swizzle),
                    use_mxfp8=bool(quant_info.use_mxfp8),
                    is_fp4_experts=bool(quant_info.is_fp4_experts),
                    activation_scale_block_size=runner_input.activation_scale_block_size,
                    layout_alignment=running_state.get("contiguous_layout_alignment"),
                ),
                allocate_output=lambda: self._allocate_down_output(
                    all_tokens, K, hidden_states_device
                ),
            )
            if cake_output is not None:
                dispose_tensor(hidden_states)
                dispose_tensor(hidden_states_scale)
                return cake_output

        recipe_a, recipe_b = quant_info.scale_recipes(
            activation_block_size=runner_input.activation_scale_block_size,
            hidden_size=K,
            activation_scale_width=hidden_states_scale.shape[-1],
        )

        w13_weight_fp8 = (
            quant_info.w13_weight,
            quant_info.w13_scale,
        )
        w2_weight_fp8 = (quant_info.w2_weight, quant_info.w2_scale)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            hidden_states_scale = tma_align_input_scale(hidden_states_scale)

        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (hidden_states, hidden_states_scale),
            w13_weight_fp8,
            gateup_output,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        if trace_deepep_v2_contig:
            torch.cuda.synchronize()
            logger.warning("DeepEP v2 expanded contig gateup GEMM returned")

        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        if self.config.activation == "situ":
            situ_beta = self.config.gemm1_alpha
            situ_linear_beta = self.config.gemm1_clamp_limit
            assert situ_beta is not None and situ_linear_beta is not None
            if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
                # Fused SiTU + per-group fp8 quant over the compacted rows,
                # then the proven round-up e8m0 cast (mn-major packed layout).
                rows = gateup_output.shape[0]
                half_n = N // 2
                kg = half_n // scale_block_size
                down_input_fp8 = torch.empty(
                    (rows, half_n),
                    device=gateup_output.device,
                    dtype=torch.float8_e4m3fn,
                )
                s = torch.empty(
                    (rows, kg), device=gateup_output.device, dtype=torch.float32
                )
                _situ_mul_quant_contig_kernel[(rows,)](
                    gateup_output,
                    down_input_fp8,
                    s,
                    half_n,
                    kg,
                    situ_beta,
                    situ_linear_beta,
                    GROUP=scale_block_size,
                    KG_POW2=triton.next_power_of_2(kg),
                    num_warps=8,
                )
                del gateup_output
                down_input_scale = _cast_to_e8m0_with_rounding_up(
                    s.unsqueeze(0)
                ).squeeze(0)
            else:
                from sglang.kernels.ops.quantization.fp8_kernel import (
                    sglang_per_token_group_quant_fp8,
                )

                gate = gateup_output[:, : N // 2].float()
                up = gateup_output[:, N // 2 :].float()
                gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
                up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
                down_input = (gate * up).to(torch.bfloat16)
                del gateup_output

                down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
                    down_input,
                    scale_block_size,
                    column_major_scales=False,
                    scale_tma_aligned=False,
                    scale_ue8m0=False,
                )
                del down_input
        elif self.use_swizzle or self.config.silu_mul_keep_fp32:
            swiglu_limit_arg: Optional[float] = self.swiglu_limit
            use_contig_swizzle = self.use_swizzle and not running_state.get(
                "deepep_v2_disable_contig_swizzle", False
            )

            down_input_fp8 = torch.empty(
                (all_tokens, N // 2),
                device=gateup_output.device,
                dtype=torch.float8_e4m3fn,
            )
            down_input_scale = create_per_token_group_quant_fp8_output_scale(
                x_shape=(all_tokens, N // 2),
                device=gateup_output.device,
                group_size=scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            silu_and_mul_contig_post_quant(
                input=gateup_output,
                output=down_input_fp8,
                output_scale=down_input_scale,
                quant_group_size=scale_block_size,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                swiglu_limit=swiglu_limit_arg,
                swizzle=use_contig_swizzle,
            )
            del gateup_output
        else:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            if not _is_musa:
                down_input = torch.empty(
                    (all_tokens, N // 2),
                    device=gateup_output.device,
                    dtype=torch.bfloat16,
                )
                if self.swiglu_limit is not None:
                    # Fuse the SwiGLU limit with the activation. The quantizing
                    # sibling only supports a group size of 128.
                    silu_and_mul_clamp(
                        gateup_output.view(-1, N), down_input, self.swiglu_limit
                    )
                else:
                    _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
            else:
                if self.swiglu_limit is not None:
                    gateup_output = _apply_swiglu_limit(
                        gateup_output, swiglu_limit=self.swiglu_limit
                    )
                down_input = _silu_and_mul_musa(gateup_output.view(-1, N))
            del gateup_output

            down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
                down_input,
                scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            del down_input
        if trace_deepep_v2_contig:
            torch.cuda.synchronize()
            logger.warning("DeepEP v2 expanded contig activation returned")

        deepep_v2_expanded = running_state.get("deepep_v2_expanded", False)
        # Folding the row weight into down_input_scale needs a non-power-of-two
        # scale; ue8m0 is a power of two, so it weights down_output before combine.
        # no_combine asks for raw rows, so the fold must not pre-apply the weight.
        fuse_weight_into_scale = (
            deepep_v2_expanded
            and not self.config.no_combine
            and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and running_state.get("topk_weights") is not None
        )
        if fuse_weight_into_scale:
            from sglang.kernels.ops.moe.ep_moe_kernels import scale_expanded_rows_

            scale_expanded_rows_(down_input_scale, running_state["topk_weights"])
            running_state["deepep_v2_weight_prefused"] = True

        down_output = self._allocate_down_output(all_tokens, K, hidden_states_device)
        if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            down_input_scale = tma_align_input_scale(down_input_scale)

        # The down activation is quantized here, independently of dispatch.
        recipe_a_down, _ = quant_info.scale_recipes(
            activation_block_size=scale_block_size,
            hidden_size=down_input_fp8.shape[-1],
            activation_scale_width=down_input_scale.shape[-1],
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
            recipe_a=recipe_a_down,
            recipe_b=recipe_b,
        )
        if trace_deepep_v2_contig:
            torch.cuda.synchronize()
            logger.warning("DeepEP v2 expanded contig down GEMM returned")

        return down_output

    def _run_bf16_contiguous_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        hidden_states = runner_input.hidden_states
        all_tokens = running_state["all_tokens"]
        hidden_states_device = running_state["hidden_states_device"]
        hidden_states_shape = running_state["hidden_states_shape"]
        m_indices = runner_input.m_indices

        N = quant_info.w13_weight.size(1)
        K = hidden_states_shape[1]

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight

        # GroupGemm-1: (M, K) (E, N, K) -> (M, N)
        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        deep_gemm_wrapper.grouped_gemm_nt_bf16_contig(
            hidden_states,
            w13_weight,
            gateup_output,
            m_indices,
        )

        dispose_tensor(hidden_states)

        # Act: (M, N) -> (M, N/2)
        if not _is_musa:
            down_input = torch.empty(
                (
                    all_tokens,
                    N // 2,
                ),
                device=gateup_output.device,
                dtype=torch.bfloat16,
            )
            _legacy_silu_and_mul(gateup_output.view(-1, N), down_input)
        else:
            down_input = _silu_and_mul_musa(gateup_output.view(-1, N))
        del gateup_output

        # GroupGemm-2: (M, N/2) (E, K, N/2) -> (M, K)
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            down_output = torch.empty(
                (all_tokens, K),
                device=hidden_states_device,
                dtype=torch.bfloat16,
            )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_contig(
            down_input,
            w2_weight,
            down_output,
            m_indices,
        )

        return down_output

    def _run_masked_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.srt.layers import deep_gemm_wrapper

        hidden_states = runner_input.hidden_states
        hidden_states_scale = runner_input.hidden_states_scale
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale

        hidden_states_device = running_state["hidden_states_device"]
        trace_deepep_v2_masked = envs.SGLANG_DEEPEP_V2_TRACE_MASKED.get()
        if trace_deepep_v2_masked:
            logger.warning(
                "DeepEP v2 masked runner enter: hidden=%s hidden_stride=%s "
                "scale=%s scale_stride=%s masked_m=%s expected_m=%s",
                tuple(hidden_states.shape),
                hidden_states.stride(),
                (
                    None
                    if hidden_states_scale is None
                    else tuple(hidden_states_scale.shape)
                ),
                None if hidden_states_scale is None else hidden_states_scale.stride(),
                masked_m.detach().cpu().tolist(),
                expected_m,
            )

        use_mxfp8 = quant_info.use_mxfp8
        scale_block_size = quant_info.block_shape[1] if quant_info.block_shape else 128

        recipe_a, recipe_b = quant_info.scale_recipes(
            activation_block_size=runner_input.activation_scale_block_size,
            hidden_size=hidden_states.shape[-1],
            activation_scale_width=hidden_states_scale.shape[-1],
        )

        # GroupGemm-0
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            if hidden_states_scale.dtype != torch.int:
                b, s_mn, s_k = hidden_states_scale.shape
                assert s_mn % 4 == 0 and s_k % 4 == 0, (
                    f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"
                )
                hidden_states_scale = _cast_to_e8m0_with_rounding_up(
                    hidden_states_scale
                )
        elif deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            hidden_states_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                hidden_states_scale
            )

        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        try:
            gateup_output = torch.empty(
                (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
            )
        except torch.OutOfMemoryError:
            logger.error(
                "Masked grouped-GEMM workspace allocation failed "
                "(num_groups=%d m=%d n=%d). If this happens under saturated "
                "dp-attention prefill, try SGLANG_OPT_DG_MASKED_M_CAP=1.",
                num_groups,
                m,
                n,
            )
            raise
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scale),
            gateup_output,
            masked_m,
            expected_m,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        if trace_deepep_v2_masked:
            torch.cuda.synchronize()
            logger.warning("DeepEP v2 masked runner gateup GEMM returned")
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)

        swiglu_limit_arg: Optional[float] = None
        if self.swiglu_limit is not None:
            swiglu_limit_arg = self.swiglu_limit

        # Act.
        if self.config.activation == "situ":
            scale_block_size = 128
            down_input, down_input_scale = _varlen_deep_gemm_situ_mul_quant(
                gateup_output,
                masked_m,
                group_size=scale_block_size,
                topk=self.config.top_k,
                beta=self.config.gemm1_alpha,
                linear_beta=self.config.gemm1_clamp_limit,
            )
        else:
            topk_ids_rs = running_state.get("topk_ids")
            num_real_tokens = (
                topk_ids_rs.shape[0]
                if (
                    use_mxfp8
                    and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                    and topk_ids_rs is not None
                    and "src2dst" in running_state
                )
                else None
            )
            down_input, down_input_scale = _varlen_deep_gemm_silu_mul_quant(
                gateup_output,
                masked_m,
                group_size=scale_block_size,
                topk=self.config.top_k,
                swiglu_limit=swiglu_limit_arg,
                swizzle=self.use_swizzle,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_clamp_limit=self.config.gemm1_clamp_limit,
                num_real_tokens=num_real_tokens,
                silu_mul_keep_fp32=self.config.silu_mul_keep_fp32,
            )
        if trace_deepep_v2_masked:
            torch.cuda.synchronize()
            logger.warning("DeepEP v2 masked runner activation returned")
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

        if (
            use_mxfp8
            and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and down_input_scale.dtype != torch.int32
        ):
            import deep_gemm.utils.layout

            down_input_scale = (
                deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                    down_input_scale
                )
            )
        elif deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
            down_input_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(
                down_input_scale
            )

        recipe_a_down, _ = quant_info.scale_recipes(
            activation_block_size=scale_block_size,
            hidden_size=down_input.shape[-1],
            activation_scale_width=down_input_scale.shape[-1],
        )
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            down_output = torch.empty(
                (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
            )

        down_gemm_overlap_args = running_state.get("down_gemm_overlap_args", None)
        if down_gemm_overlap_args is None:
            gemm_overlap_args_dict = {}
        else:
            down_gemm_overlap_args.start_event.record()
            max_block_n = (
                160 if (_DEEPGEMM_ON_H20 and runner_input.expected_m <= 64) else 256
            )
            gemm_overlap_args_dict = {
                "overlap_args": down_gemm_overlap_args,
                "max_block_n": max_block_n,
            }

        deep_gemm_return_value = deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (down_input, down_input_scale),
            (w2_weight, w2_scale),
            down_output,
            masked_m,
            expected_m,
            recipe_a=recipe_a_down,
            recipe_b=recipe_b,
            **gemm_overlap_args_dict,
        )
        if trace_deepep_v2_masked:
            torch.cuda.synchronize()
            logger.warning("DeepEP v2 masked runner down GEMM returned")
        meta_overlap_args = running_state.get("meta_overlap_args", None)
        # Returns (block_m, threshold) only with down-gemm overlap, else None;
        # meta_overlap_args may be set without overlap, so guard the unpack.
        if meta_overlap_args is not None and deep_gemm_return_value is not None:
            block_m, threshold = deep_gemm_return_value
            meta_overlap_args["block_m"] = block_m
            meta_overlap_args["threshold"] = threshold

        return down_output

    def _run_masked_bf16_gemm(
        self,
        runner_input: DeepGemmRunnerInput,
        quant_info: DeepGemmMoeQuantInfo,
        running_state: dict,
    ) -> torch.Tensor:
        from sglang.kernels.ops.moe.ep_moe_kernels import silu_and_mul_masked_fwd
        from sglang.srt.layers import deep_gemm_wrapper

        hidden_states = runner_input.hidden_states
        masked_m = runner_input.masked_m
        expected_m = runner_input.expected_m

        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight

        hidden_states_device = running_state["hidden_states_device"]

        # GroupGemm-0
        num_groups, m, k = hidden_states.shape
        n = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_masked(
            hidden_states,
            w13_weight,
            gateup_output,
            masked_m,
            expected_m,
        )
        dispose_tensor(hidden_states)

        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        # Act
        silu_and_mul_masked_fwd(gateup_output, down_input, masked_m)
        del gateup_output

        # GroupGemm-1
        n = w2_weight.shape[1]

        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            down_output = torch.empty(
                (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
            )
        deep_gemm_wrapper.grouped_gemm_nt_bf16_masked(
            down_input,
            w2_weight,
            down_output,
            masked_m,
            expected_m,
        )
        # Note: BF16 masked gemm doesn't support overlap_args, so no return value unpack

        return down_output

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.DEEP_GEMM


@register_pre_permute("standard", "deep_gemm")
def pre_permute_standard_to_deep_gemm(
    dispatch_output: StandardDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
    expert_start: int = 0,
) -> DeepGemmRunnerInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import (
        ep_scatter,
        fused_moe_dispatch_index,
        moe_ep_deepgemm_preprocess,
    )

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output
    # SM120's DeepGEMM grouped GEMM consumes standard-layout activations only.
    # Feeding it the shared masked/swizzled layout does not raise -- it silently
    # returns wrong results (GSM8K 0.96 -> 0.06, measured on 4x RTX 6000D), so
    # refuse the combination rather than corrupt output.
    assert deep_gemm_sm120.is_supported(), (
        "--moe-runner-backend deep_gemm on consumer Blackwell (SM120) requires "
        "the standard-layout MoE path, which is unavailable in this build."
    )
    sm120_input = deep_gemm_sm120.maybe_pre_permute(
        hidden_states, topk_ids, topk_weights, quant_info, runner_config, running_state
    )
    if sm120_input is not None:
        return sm120_input

    hidden_states_shape = hidden_states.shape
    hidden_states_dtype = hidden_states.dtype
    hidden_states_device = hidden_states.device
    hidden_states_ref = hidden_states

    topk_weights, topk_ids = topk_weights, topk_ids

    if (
        deep_gemm_sm120.allows_masked_standard_layout()
        and _should_use_masked_standard_layout(runner_config, quant_info, hidden_states)
    ):
        output_dtype = (
            torch.bfloat16
            if quant_info.w13_weight.dtype == torch.bfloat16
            else torch.float8_e4m3fn
        )
        masked_m, _, src2dst, hidden_states, hidden_states_scale = (
            moe_ep_deepgemm_preprocess(
                topk_ids,
                runner_config.num_local_experts,
                hidden_states,
                runner_config.top_k,
                quant_info.block_shape,
                output_dtype=output_dtype,
                use_mxfp8=quant_info.use_mxfp8,
                expert_start=expert_start,
            )
        )
        # Use the global expert count because expected_m is a tuning hint, not
        # the per-rank buffer capacity.
        expected_m = max(
            1,
            ceil_div(
                hidden_states_shape[0] * runner_config.top_k,
                runner_config.num_experts,
            ),
        )

        if runner_config.inplace:
            dispose_tensor(hidden_states_ref)

        running_state["topk_ids"] = topk_ids
        running_state["topk_weights"] = topk_weights
        running_state["hidden_states_shape"] = hidden_states_shape
        running_state["hidden_states_dtype"] = hidden_states_dtype
        running_state["hidden_states_device"] = hidden_states_device
        running_state["src2dst"] = src2dst
        return DeepGemmRunnerInput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            use_masked_gemm=True,
            masked_m=masked_m,
            expected_m=expected_m,
            activation_scale_block_size=(
                quant_info.block_shape[1] if quant_info.block_shape else 128
            ),
        )

    # The compact layout avoids scaling masked buffers with the expert count.
    # Scatter and post-permute skip non-local experts mapped to -1.
    num_experts = runner_config.num_local_experts
    num_assignments = topk_ids.numel()
    block_e = deep_gemm_wrapper.get_contiguous_layout_alignment(
        num_assignments, num_experts
    )
    all_tokens = _get_compact_all_tokens(num_assignments, num_experts, block_e)

    tokens_per_expert, unused_masked_dst = fused_moe_dispatch_index(
        topk_ids, num_experts, 1, expert_start=expert_start
    )
    dispose_tensor(unused_masked_dst)
    valid_tokens_per_expert = tokens_per_expert
    if _is_cuda:
        tokens_per_expert = pad_expert_counts(tokens_per_expert, block_e, all_tokens)
    else:
        # The Triton kernel is CUDA-only. Keep the existing MUSA-compatible
        # tensor implementation for other DeepGEMM backends.
        tokens_per_expert = (ceil_div(tokens_per_expert, block_e) * block_e).to(
            torch.int32
        )
        tokens_per_expert[-1].add_(all_tokens - tokens_per_expert.sum())

    k = hidden_states.size(1)
    output_dtype = (
        torch.bfloat16
        if quant_info.w13_weight.dtype == torch.bfloat16
        else torch.float8_e4m3fn
    )
    if output_dtype == torch.bfloat16:
        packed_input_source = hidden_states
        packed_input_source_scale = None
        packed_input = torch.empty(
            (all_tokens, k), device=hidden_states_device, dtype=torch.bfloat16
        )
        # ep_scatter ignores scales for BF16, but a real tensor keeps its
        # Triton signature uniform across the existing DeepEP caller.
        packed_input_scale = torch.empty(
            (all_tokens, 1), device=hidden_states_device, dtype=torch.float32
        )
    else:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        block_k = quant_info.block_shape[1] if quant_info.block_shape else 128
        packed_input_source, packed_input_source_scale = (
            sglang_per_token_group_quant_fp8(
                hidden_states,
                block_k,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
        )
        # ep_scatter writes every live row and the grouped GEMM's results for
        # the alignment padding are dropped by post_reorder, so the zeroing is
        # dead work -- 174 MB per layer at bs=64. The sibling dispatch path in
        # this file already allocates its equivalent buffer with torch.empty
        # unless deterministic inference is on; match it.
        deterministic = get_exec().deterministic.enable_deterministic_inference
        packed_input = (torch.zeros if deterministic else torch.empty)(
            (all_tokens, k),
            device=hidden_states_device,
            dtype=torch.float8_e4m3fn,
        )
        scale_width = k // block_k
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            scale_width = ceil_div(scale_width, 4)
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            packed_input_scale = torch.zeros(
                (scale_width, all_tokens),
                device=hidden_states_device,
                dtype=packed_input_source_scale.dtype,
            ).transpose(0, 1)
        else:
            packed_input_scale = torch.zeros(
                (all_tokens, scale_width),
                device=hidden_states_device,
                dtype=packed_input_source_scale.dtype,
            )

    expert_start_loc = torch.empty(
        num_experts, device=hidden_states_device, dtype=torch.int32
    )
    m_indices = torch.empty(all_tokens, device=hidden_states_device, dtype=torch.int32)
    src2dst = torch.empty_like(topk_ids, dtype=torch.int32)
    ep_scatter(
        packed_input_source,
        packed_input_source_scale,
        topk_ids,
        tokens_per_expert,
        valid_tokens_per_expert,
        expert_start_loc,
        packed_input,
        packed_input_scale,
        m_indices,
        src2dst,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        quant_block_size=(quant_info.block_shape[1] if quant_info.block_shape else 128),
        expert_alignment=block_e,
        expert_start=expert_start,
    )
    if packed_input_source is not hidden_states:
        dispose_tensor(packed_input_source)
    if packed_input_source_scale is not None:
        dispose_tensor(packed_input_source_scale)

    # Preserve the input when a shared expert or its gate may still use it.
    if runner_config.inplace:
        dispose_tensor(hidden_states_ref)

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["hidden_states_device"] = hidden_states_device
    running_state["src2dst"] = src2dst
    running_state["all_tokens"] = all_tokens
    running_state["contiguous_layout_alignment"] = block_e

    return DeepGemmRunnerInput(
        hidden_states=packed_input,
        hidden_states_scale=packed_input_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
        activation_scale_block_size=(
            quant_info.block_shape[1] if quant_info.block_shape else 128
        ),
    )


@register_pre_permute("flashinfer", "deep_gemm")
def pre_permute_flashinfer_to_deep_gemm(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    """Feed one-sided A2A output into DeepGEMM with fused expert remapping."""

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

    if dispatch_output.hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "FlashInfer A2A + DeepGEMM requires a BF16 dispatch payload, got "
            f"{dispatch_output.hidden_states.dtype}."
        )
    if dispatch_output.hidden_states_scale is not None:
        raise ValueError(
            "FlashInfer A2A + DeepGEMM expects unquantized BF16 dispatch; "
            "hidden_states_scale must be None."
        )
    if dispatch_output.topk_output.topk_ids.dtype != torch.int32:
        raise TypeError(
            "FlashInfer A2A expert IDs must be int32 before DeepGEMM, got "
            f"{dispatch_output.topk_output.topk_ids.dtype}."
        )

    standard_output = StandardDispatchOutput(
        hidden_states=dispatch_output.hidden_states,
        hidden_states_scale=None,
        topk_output=dispatch_output.topk_output,
    )
    expert_start = get_parallel().moe_ep_rank * runner_config.num_local_experts
    return pre_permute_standard_to_deep_gemm(
        standard_output,
        quant_info,
        runner_config,
        running_state,
        expert_start=expert_start,
    )


@register_post_permute("deep_gemm", "standard")
def post_permute_deep_gemm_to_standard(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    sm120_output = deep_gemm_sm120.maybe_post_permute(
        runner_output, runner_config, running_state
    )
    if sm120_output is not None:
        return sm120_output

    hidden_states_shape = running_state["hidden_states_shape"]
    hidden_states_dtype = running_state["hidden_states_dtype"]
    hidden_states_device = running_state["hidden_states_device"]
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]

    src2dst = running_state["src2dst"]

    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
    post_reorder_deepgemm(
        runner_output.hidden_states,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        runner_config.top_k,
        hidden_states_shape[0],
        hidden_states_shape[1],
        (
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
    )
    dispose_tensor(runner_output.hidden_states)

    return StandardCombineInput(
        hidden_states=output,
    )


@register_post_permute("deep_gemm", "flashinfer")
def post_permute_deep_gemm_to_flashinfer(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    """Reuse DeepGEMM's weighted post-permute and hand BF16 to A2A combine."""

    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )

    standard_input = post_permute_deep_gemm_to_standard(
        runner_output, quant_info, runner_config, running_state
    )
    if standard_input.hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "FlashInfer A2A + DeepGEMM combine payload must be BF16, got "
            f"{standard_input.hidden_states.dtype}."
        )
    return FlashinferCombineInput(hidden_states=standard_input.hidden_states)


@register_pre_permute("deepep_ll", "deep_gemm")
def pre_permute_deepep_ll_to_deep_gemm(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, expected_m = (
        dispatch_output
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["hidden_states_device"] = hidden_states.device
    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
        activation_scale_block_size=128,
    )


@register_post_permute("deep_gemm", "deepep_ll")
def post_permute_deep_gemm_to_deepep_ll(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_normal", "deep_gemm")
def pre_permute_deepep_normal_to_deep_gemm(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    assert runner_config.activation in ("silu", "situ")

    all_tokens = sum(num_recv_tokens_per_expert)
    running_state["all_tokens"] = all_tokens

    K = hidden_states.shape[1]

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    hidden_states_dtype = hidden_states.dtype

    running_state["hidden_states_shape"] = hidden_states_shape
    running_state["hidden_states_device"] = hidden_states_device
    running_state["hidden_states_dtype"] = hidden_states_dtype
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    # Deterministic inference zero-fills the scatter buffers: expert-alignment
    # padding leaves slots that ep_scatter never writes, and pad garbage in
    # input_tensor would leak batch-dependent values into the grouped GEMM.
    # The scale buffer only matters for FP8 activations sharing this
    # pre-permute (ep_scatter skips scales entirely for BF16 dispatch).
    deterministic = get_exec().deterministic.enable_deterministic_inference
    buffer_init = torch.zeros if deterministic else torch.empty

    input_tensor = buffer_init(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        # TODO check whether need `zeros`
        input_tensor_scale = torch.zeros(
            (ceil_div(K // 128, 4), all_tokens),
            device=hidden_states.device,
            dtype=torch.int,
        ).transpose(0, 1)
    else:
        input_tensor_scale = buffer_init(
            (all_tokens, K // 128),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    m_indices = buffer_init(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)

    if get_offloader().forbid_copy_engine_usage:
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
            num_recv_tokens_per_expert
        )
    else:
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

    ep_scatter(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        num_recv_tokens_per_expert_gpu,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )
    dispose_tensor(hidden_states)
    if hidden_states_scale is not None:
        dispose_tensor(hidden_states_scale)

    running_state["output_index"] = output_index

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
        activation_scale_block_size=128,
    )


@register_post_permute("deep_gemm", "deepep_normal")
def post_permute_deep_gemm_to_deepep_normal(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    hidden_states = runner_output.hidden_states
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    output_index = running_state["output_index"]

    gather_out = torch.empty(
        running_state["hidden_states_shape"],
        device=running_state["hidden_states_device"],
        dtype=torch.bfloat16,
    )
    ep_gather(hidden_states, topk_ids, topk_weights, output_index, gather_out)

    return DeepEPNormalCombineInput(
        hidden_states=gather_out,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


def _varlen_deep_gemm_situ_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: torch.Tensor,
    group_size: int,
    topk: int,
    beta: float,
    linear_beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiTU activation + per-group fp8 quant via CUDA JIT kernel."""
    from sglang.kernels.ops.moe import situ_and_mul_masked_post_quant

    E, N, D_2 = gateup_output.shape
    D = D_2 // 2
    G = D // group_size
    packed_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0

    down_input = torch.empty(
        (E, N, D), device=gateup_output.device, dtype=torch.float8_e4m3fn
    )
    if packed_ue8m0:
        down_input_scale = torch.empty(
            (E, G // 4, N), device=gateup_output.device, dtype=torch.int32
        )
    else:
        down_input_scale = torch.empty(
            (E, N, G), device=gateup_output.device, dtype=torch.float32
        )

    situ_and_mul_masked_post_quant(
        gateup_output,
        down_input,
        down_input_scale,
        group_size,
        masked_m,
        beta=beta,
        linear_beta=linear_beta,
        scale_ue8m0=packed_ue8m0,
        topk=topk,
        transposed=packed_ue8m0,
    )

    if packed_ue8m0:
        down_input_scale = down_input_scale.transpose(-1, -2)

    return down_input, down_input_scale


def _varlen_deep_gemm_silu_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: Optional[torch.Tensor],
    group_size: int,
    topk: int,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
    num_real_tokens: Optional[int] = None,
    silu_mul_keep_fp32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert masked_m is not None
    hidden_states_device = gateup_output.device
    E, N, D_2 = gateup_output.shape
    D = D_2 // 2
    del D_2
    G = D // group_size

    if silu_mul_keep_fp32 and gemm1_alpha is not None:
        raise ValueError("silu_mul_keep_fp32 does not support gemm1_alpha")

    # oai-swiglu (gemm1_alpha) stays on the Triton kernel until
    # per_token_group_quant grows an activation-kind axis. The output_scale dtype picks the schedule: packed
    # int32 UE8M0 (no follow-up transform; needs G % 4 == 0 and the
    # num_real_tokens grid bound) when eligible, row-major fp32 otherwise.
    if gemm1_alpha is not None:
        assert swiglu_limit is None, (
            "swiglu_limit and gemm1_alpha are mutually exclusive"
        )
        assert not swizzle, "swizzle is not supported with gemm1_alpha"
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_and_mul_masked_post_quant_fwd,
        )

        use_packed = (
            deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and num_real_tokens is not None
            and G % 4 == 0
            and D % (group_size * 4) == 0
        )
        down_input = torch.empty(
            (E, N, D), device=hidden_states_device, dtype=torch.float8_e4m3fn
        )
        down_input_scale = torch.empty(
            (E, G // 4, N) if use_packed else (E, N, G),
            device=hidden_states_device,
            dtype=torch.int32 if use_packed else torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            group_size,
            masked_m,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit or 0.0,
            num_real_tokens=num_real_tokens,
            topk=topk,
        )
        if use_packed:
            down_input_scale = down_input_scale.transpose(-1, -2)
        return down_input, down_input_scale

    # Only explicit precision requests opt additional callers into this kernel;
    # the generic fused quantizer rounds its SiLU intermediates to BF16.
    if swiglu_limit is not None or swizzle or silu_mul_keep_fp32:
        assert N % 4 == 0 and G % 4 == 0 and D // 8 >= E, (
            "DSV4 JIT activation requires N % 4 == 0, G % 4 == 0 and "
            f"D // 8 >= num_experts, got N={N} G={G} D={D} E={E}"
        )
        packed_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        down_input = torch.empty(
            (E, N, D), device=hidden_states_device, dtype=torch.float8_e4m3fn
        )
        down_input_scale = torch.empty(
            (E, G // 4, N) if packed_ue8m0 else (E, N, G),
            device=hidden_states_device,
            dtype=torch.int32 if packed_ue8m0 else torch.float32,
        )
        silu_and_mul_masked_post_quant(
            gateup_output,
            down_input,
            down_input_scale,
            group_size,
            masked_m,
            scale_ue8m0=packed_ue8m0,
            topk=topk,
            transposed=packed_ue8m0,
            swiglu_limit=swiglu_limit,
            swizzle=swizzle,
        )
        if packed_ue8m0:
            down_input_scale = down_input_scale.transpose(-1, -2)
        return down_input, down_input_scale

    # Default plain-silu path: the unified JIT masked fused quant. It allocates
    # the outputs itself, with scales directly in the layout deep_gemm consumes
    # (packed-int32 col-major for UE8M0, TMA-aligned col-major fp32 otherwise),
    # so the caller's get_mn_major transform short-circuits.
    expected_m = ceil_div(num_real_tokens * topk, E) if num_real_tokens else None
    return per_token_group_quant(
        gateup_output,
        group_size=group_size,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        fuse_silu_and_mul=True,
        masked_m=masked_m,
        expected_m=expected_m,
        column_major_scales=True,
    )


@triton.jit
def _situ_mul_quant_contig_kernel(
    g_ptr,  # [rows, 2N] bf16, non-interleaved [gate; up] halves
    q_ptr,  # [rows, N] fp8 out
    s_ptr,  # [rows, KG] fp32 scales out
    N,
    KG,
    situ_beta,
    situ_linear_beta,
    GROUP: tl.constexpr,
    KG_POW2: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rows2d = tl.arange(0, KG_POW2)[:, None]
    cols = tl.arange(0, GROUP)[None, :]
    offs = rows2d * GROUP + cols
    mask = rows2d < KG
    gate = tl.load(g_ptr + row * 2 * N + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(g_ptr + row * 2 * N + N + offs, mask=mask, other=0.0).to(tl.float32)
    # tanh(x) == 2*sigmoid(2x) - 1 (avoids a libdevice dependency)
    gate_t = 2.0 * tl.sigmoid(2.0 * gate / situ_beta) - 1.0
    gate = situ_beta * gate_t * tl.sigmoid(gate)
    up_t = 2.0 * tl.sigmoid(2.0 * up / situ_linear_beta) - 1.0
    y = gate * situ_linear_beta * up_t
    amax = tl.clamp(tl.max(tl.abs(y), axis=1), min=1e-10, max=float("inf"))
    q = (y * (448.0 / amax)[:, None]).to(tl.float8e4nv)
    tl.store(q_ptr + row * N + offs, q, mask=mask)
    srow = tl.arange(0, KG_POW2)
    tl.store(s_ptr + row * KG + srow, amax / 448.0, mask=srow < KG)


def _apply_swiglu_limit(
    gateup_output: torch.Tensor, swiglu_limit: float
) -> torch.Tensor:
    """Clamp the contiguous runner's owned GEMM workspace in place."""
    assert swiglu_limit == 10

    num_tokens, hidden_size_x2 = gateup_output.shape
    assert gateup_output.dtype == torch.bfloat16

    gate, up = torch.chunk(gateup_output, chunks=2, dim=-1)
    assert gate.shape == (num_tokens, hidden_size_x2 // 2)
    assert up.shape == (num_tokens, hidden_size_x2 // 2)

    # Both halves are views of a fresh GEMM output. Avoid separate clamped
    # copies and their concatenation: large compact prefills need that
    # headroom for the activation and down-projection workspaces.
    up.clamp_(min=-swiglu_limit, max=swiglu_limit)
    gate.clamp_(max=swiglu_limit)
    return gateup_output


@register_pre_permute("deepep_v2", "deep_gemm")
def pre_permute_deepep_v2_to_deep_gemm(
    dispatch_output: DeepEPv2DispatchOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import (
        ep_scatter_from_psum,
        fill_m_indices_from_psum,
    )

    hidden_states = dispatch_output.hidden_states
    hidden_states_scale = dispatch_output.hidden_states_scale
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights
    psum_num_recv_tokens_per_expert = dispatch_output.psum_num_recv_tokens_per_expert
    is_expanded = dispatch_output.is_expanded
    deepep_v2_use_masked = dispatch_output.use_masked_gemm
    deepep_v2_expected_m = dispatch_output.expected_m
    deepep_v2_masked_max_m = dispatch_output.masked_max_m
    deepep_v2_total_expanded = dispatch_output.total_expanded
    deepep_v2_expert_alignment = dispatch_output.expert_alignment
    is_fp8 = hidden_states_scale is not None
    if not is_fp8 and hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(
            "DeepEP v2 -> DeepGEMM requires either FP8 dispatch output with "
            "activation scales or BF16 dispatch output, but the dispatch "
            f"output carried {hidden_states.dtype} without scales."
        )
    assert runner_config.activation == "silu"

    if is_expanded:
        if psum_num_recv_tokens_per_expert is None:
            raise RuntimeError(
                "DeepEP v2 requires the native expert prefix sums from the "
                "ElasticBuffer dispatch handle."
            )
        all_tokens = hidden_states.shape[0]
        running_state["all_tokens"] = all_tokens
        running_state["hidden_states_shape"] = hidden_states.shape
        running_state["hidden_states_device"] = hidden_states.device
        running_state["hidden_states_dtype"] = hidden_states.dtype
        running_state["topk_ids"] = None
        running_state["topk_weights"] = topk_weights
        running_state["deepep_v2_expanded"] = True

        if deepep_v2_use_masked:
            # masked_m bounds each expert independently of buffer capacity.
            from sglang.kernels.ops.moe.ep_moe_kernels import expand_to_masked_slab

            num_local_experts = psum_num_recv_tokens_per_expert.shape[0]
            input_tensor, input_tensor_scale, masked_m = expand_to_masked_slab(
                hidden_states,
                hidden_states_scale,
                psum_num_recv_tokens_per_expert,
                num_local_experts,
                deepep_v2_masked_max_m,
                deepep_v2_expert_alignment,
            )
            running_state["deepep_v2_masked"] = True
            running_state["deepep_v2_psum"] = psum_num_recv_tokens_per_expert
            running_state["deepep_v2_total_expanded"] = deepep_v2_total_expanded
            running_state["deepep_v2_expert_alignment"] = deepep_v2_expert_alignment
            return DeepGemmRunnerInput(
                hidden_states=input_tensor,
                hidden_states_scale=input_tensor_scale,
                use_masked_gemm=True,
                masked_m=masked_m,
                expected_m=deepep_v2_expected_m,
                activation_scale_block_size=(
                    dispatch_output.activation_scale_block_size
                ),
            )

        num_local_experts = psum_num_recv_tokens_per_expert.shape[0]
        m_indices = fill_m_indices_from_psum(
            psum_num_recv_tokens_per_expert,
            num_local_experts,
            all_tokens,
            deepep_v2_expert_alignment,
        )
        return DeepGemmRunnerInput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            use_masked_gemm=False,
            m_indices=m_indices,
            activation_scale_block_size=dispatch_output.activation_scale_block_size,
        )

    all_tokens = int(psum_num_recv_tokens_per_expert[-1].item())
    K = hidden_states.shape[1]
    scale_block_size = dispatch_output.activation_scale_block_size
    running_state["all_tokens"] = all_tokens
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_device"] = hidden_states.device
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    input_tensor = torch.empty(
        (all_tokens, K), device=hidden_states.device, dtype=hidden_states.dtype
    )
    if not is_fp8:
        input_tensor_scale = None
    elif deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        # Packed UE8M0 scales require zero padding lanes.
        input_tensor_scale = torch.zeros(
            (ceil_div(K // scale_block_size, 4), all_tokens),
            device=hidden_states.device,
            dtype=torch.int,
        ).transpose(0, 1)
    else:
        input_tensor_scale = torch.empty(
            (all_tokens, K // scale_block_size),
            device=hidden_states.device,
            dtype=torch.float32,
        )
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)
    # Contiguous psum already includes the 128-row expert alignment.
    expert_start_loc = torch.empty_like(psum_num_recv_tokens_per_expert)
    ep_scatter_from_psum(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        psum_num_recv_tokens_per_expert,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        quant_block_size=scale_block_size,
    )
    dispose_tensor(hidden_states)
    if hidden_states_scale is not None:
        dispose_tensor(hidden_states_scale)
    running_state["output_index"] = output_index

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
        activation_scale_block_size=dispatch_output.activation_scale_block_size,
    )


@register_post_permute("deep_gemm", "deepep_v2")
def post_permute_deep_gemm_to_deepep_v2(
    runner_output: DeepGemmRunnerOutput,
    quant_info: DeepGemmMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPv2CombineInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather
    from sglang.srt.layers.moe.token_dispatcher.base import RoutewiseLayout
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import DeepEPv2CombineInput

    return_unweighted_routes = runner_config.no_combine
    if running_state.get("deepep_v2_expanded", False):
        hidden_states = runner_output.hidden_states
        topk_weights = running_state["topk_weights"]
        if running_state.get("deepep_v2_masked", False):
            # A routewise finalizer must run before router weighting. Preserve
            # one raw row per route and carry its 1-D weight to that finalizer.
            from sglang.kernels.ops.moe.ep_moe_kernels import masked_slab_to_expand

            output_capacity = running_state["deepep_v2_total_expanded"]
            if topk_weights.ndim != 1 or topk_weights.shape[0] < output_capacity:
                raise ValueError(
                    "DeepEP v2 expanded output exceeds router-weight capacity"
                )
            hidden_states = masked_slab_to_expand(
                hidden_states,
                running_state["deepep_v2_psum"],
                output_capacity,
                running_state["deepep_v2_expert_alignment"],
                topk_weights=None if return_unweighted_routes else topk_weights,
            )
            if not return_unweighted_routes:
                return DeepEPv2CombineInput(hidden_states, None)
            # Match the communication-capacity weights to the output slab.
            return DeepEPv2CombineInput(
                hidden_states=hidden_states,
                topk_weights=topk_weights[: hidden_states.shape[0]],
                routewise_layout=RoutewiseLayout.EXPANDED,
            )
        if return_unweighted_routes:
            return DeepEPv2CombineInput(
                hidden_states, topk_weights, RoutewiseLayout.EXPANDED
            )
        if topk_weights is not None and not running_state.get(
            "deepep_v2_weight_prefused", False
        ):
            # Expanded combine does not consume top-k weights;
            # skip when fold-into-scale already applied them before down_proj.
            # In-place with fp32 weights rounds once; casting the weights to
            # bf16 first costs measurable accuracy.
            hidden_states.mul_(topk_weights.unsqueeze(-1))
        return DeepEPv2CombineInput(hidden_states, None)

    hidden_states = runner_output.hidden_states
    topk_ids = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    output_index = running_state["output_index"]
    if return_unweighted_routes:
        # Restore the route dimension required by a routewise finalizer.
        # output_index maps each received token/expert slot back to the compact
        # expert-sorted DeepGEMM output; -1 denotes a non-local route.
        valid = output_index >= 0
        if hidden_states.shape[0] == 0:
            route_out = hidden_states.new_zeros(
                (*output_index.shape, hidden_states.shape[-1])
            )
        else:
            safe_output_index = output_index.clamp_min(0).to(torch.int64)
            route_out = hidden_states[safe_output_index]
            route_out.masked_fill_(~valid.unsqueeze(-1), 0)
        return DeepEPv2CombineInput(
            hidden_states=route_out,
            topk_weights=topk_weights,
            routewise_layout=RoutewiseLayout.TOKEN_TOPK,
        )
    gather_out = torch.empty(
        running_state["hidden_states_shape"],
        device=running_state["hidden_states_device"],
        dtype=torch.bfloat16,
    )
    ep_gather(hidden_states, topk_ids, topk_weights, output_index, gather_out)
    return DeepEPv2CombineInput(
        hidden_states=gather_out,
        topk_weights=topk_weights,
    )
