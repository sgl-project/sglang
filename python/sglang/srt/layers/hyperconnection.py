from sglang.kernels.ops.extend_fastpath import HC_ENABLED as _Q38C_EXTEND_PATH

import logging
import os
from functools import lru_cache
from typing import Optional

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.gemm.hc_mix import fused_hc_mix, fused_hc_mix_supported
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel, new_forward_helper_stream

logger = logging.getLogger(__name__)

# SGLANG_HC_AR_COMBINE_NORM=1: at TP > 1, a decode write-back whose block output
# owes its TP sum completes it in the kernel that applies the side-stream gate and
# the next read's norm (kernels/ops/communication/hc_all_reduce_combine_norm.py).
_HC_AR_COMBINE_NORM = os.environ.get("SGLANG_HC_AR_COMBINE_NORM", "0") == "1"
# Bench aid: log every fused write-back built (each graph capture runs Python once).
_HC_AR_VERBOSE = os.environ.get("SGLANG_HC_AR_COMBINE_NORM_VERBOSE", "0") == "1"
_HC_AR_LOGGED: set = set()
_HC_MACRO = envs.SGLANG_HC_MACRO.get()
_HC_MACRO_MIN_ROWS = max(1, envs.SGLANG_HC_MACRO_MIN_ROWS.get())
_HC_MACRO_MAX_ROWS = 32
_SPLITK_MIX_MAX_ROWS = 32 if _HC_MACRO else 24
_SPLITK_MIX_WIDE_MAX_ROWS = envs.SGLANG_HC_MIX_SPLITK_MAX_ROWS.get() if _HC_MACRO else 0
if _SPLITK_MIX_WIDE_MAX_ROWS > 0:
    from sglang.kernels.ops.gemm.hc_mix_splitk import WIDE_MAX_ROWS

    _SPLITK_MIX_WIDE_MAX_ROWS = min(_SPLITK_MIX_WIDE_MAX_ROWS, WIDE_MAX_ROWS)

_FUSED_COMBINE_MIN_ROWS = 33
_FUSED_COMBINE_MAX_ROWS = min(envs.SGLANG_HC_FUSED_COMBINE_MAX_ROWS.get(), 128)
# One launch per HC boundary at 33..N write-back rows (N <= 320; kernels/ops/gemm/hc_boundary_ks.py): the extend
# write-back and the next read's norm, gate totals and mix, K-sliced over 40 CTAs per 64 rows. Not bitwise to the
# separate kernels. 0 (default) keeps them. _FLAGS: hc_boundary_ks.FLAG_* bits (1 L2 discard, 2 tanh gate,
# 4 always write the normalized streams).
_KS_BOUNDARY_MAX_ROWS = (
    min(int(os.environ.get("SGLANG_HC_BOUNDARY_KS_MAX_ROWS", "0")), 320)
    if _HC_MACRO and _Q38C_EXTEND_PATH
    else 0
)
_KS_BOUNDARY_FLAGS = int(os.environ.get("SGLANG_HC_BOUNDARY_KS_FLAGS", "3"))
_KS_BOUNDARY_BUILT: list = []


def _hc_boundary_ks_ready() -> bool:
    """Build (or load) the one-launch boundary and its scratch once per process, before graph capture; a
    failure keeps the separate kernels."""
    if not _KS_BOUNDARY_BUILT:
        try:
            from sglang.kernels.ops.gemm.hc_boundary_ks import warmup

            warmup()
            _KS_BOUNDARY_BUILT.append(True)
        except Exception as e:  # keep the separate kernels
            logger.warning("hc_boundary_ks unavailable (%s); separate HC kernels", e)
            _KS_BOUNDARY_BUILT.append(False)
    return _KS_BOUNDARY_BUILT[0]
# Extend-size HC boundary (read once): from _HC_EXT_MIN_ROWS rows each
# write-back runs one kernel that combines, normalizes the streams for the next
# read and makes that read's gate dot totals, replacing hc_combine plus the
# next read's grouped_gemma_rmsnorm (and the gate's re-read of the normalized
# streams). Bit-identical to the stock chain. SGLANG_X_HC_EXTEND_DISABLE=1
# restores the stock extend boundary exactly.
_HC_EXT = _Q38C_EXTEND_PATH
_HC_EXT_MIN_ROWS = max(
    _HC_MACRO_MAX_ROWS + 1, int(os.environ.get("SGLANG_X_HC_EXTEND_MIN_ROWS", "1024"))
)
# Build outcome per (hc, hidden, dtype), success or failure, decided once per
# process: a failing build is not retried by every GatedResidual, and every
# layer agrees on whether the extend boundary (and its [rows, hc] partials) is
# in use.
_HC_EXT_BUILT: dict = {}
# Row bands ("verify" below 1024 rows, "prefill" from 1024) for which the extend
# boundary has logged its first use in this process.
_HC_EXT_LOGGED: set = set()


def _log_hc_ext_once(rows: int, has_partials: bool, norm: bool, gate: bool) -> None:
    band = "prefill" if rows >= 1024 else "verify"
    if band in _HC_EXT_LOGGED:
        return
    _HC_EXT_LOGGED.add(band)
    logger.info(
        "HC extend boundary engaged (%s rows): rows=%d min_rows=%d partials=%s norm=%s gate=%s",
        band, rows, _HC_EXT_MIN_ROWS, has_partials, norm, gate,
    )


def _hc_ext_ready(hc_count: int, hidden_size: int, dtype: torch.dtype) -> bool:
    key = (hc_count, hidden_size, dtype)
    ready = _HC_EXT_BUILT.get(key)
    if ready is None:
        try:
            from sglang.kernels.ops.elementwise.hc_combine_extend import warmup

            warmup(hc_count, hidden_size, dtype)
            ready = True
        except Exception as e:  # keep the stock boundary
            logger.warning("hc_combine_extend unavailable (%s); stock HC", e)
            ready = False
        _HC_EXT_BUILT[key] = ready
    return ready


@lru_cache(None)
def _hc_gate_stream(device: int) -> torch.cuda.Stream:
    return new_forward_helper_stream(device)


_BF16_GEMM_PINS = bool(envs.SGLANG_BF16_GEMM_PIN_FILE.get())


@lru_cache(None)
def _pinned_mix_compute():
    """torch.compile of GatedResidual's _mix_compute with its two GEMMs on the BF16 dispatch op, which serves
    SGLANG_BF16_GEMM_PIN_FILE's pins; the ops, and so the pointwise kernels between the GEMMs, are _mix_compute's."""
    from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

    def mix_compute(
        hyper_input_normed: torch.Tensor,
        input_mix_weight_down: torch.Tensor,
        input_mix_weight_up: torch.Tensor,
        hc: int,
        hs: int,
    ) -> torch.Tensor:
        input_mix_weight = F.silu(
            bf16_gemm_dispatch(hyper_input_normed, input_mix_weight_down, None) / hc
        )
        input_mix_weight = bf16_gemm_dispatch(input_mix_weight, input_mix_weight_up, None)
        input_mix_weight = torch.sigmoid(input_mix_weight)
        input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
        output = (
            input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
        ).mean(dim=-2)
        return output

    return torch.compile(mix_compute)


def _mix_gemms_pinned(lowrank: int, width: int) -> bool:
    if not _BF16_GEMM_PINS:
        return False
    from sglang.srt.layers import bf16_gemm_pins

    return bf16_gemm_pins.has_shape(lowrank, width) or bf16_gemm_pins.has_shape(width, lowrank)


class HyperConnectionConfig(msgspec.Struct, frozen=True):
    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, group_size: Optional[int] = None
    ):
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight.weight_loader = self._weight_loader
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(
                x, self.weight, self._jit_group_size, self.variance_epsilon
            )
        input_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.variance_epsilon)
        else:
            x_grouped = x_float.reshape(
                *x_float.shape[:-1],
                x_float.shape[-1] // self.group_size,
                self.group_size,
            )
            variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
            x_norm = (
                x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (x_norm * (1.0 + self.weight.float())).to(input_dtype)


class HyperConnectionBase(nn.Module):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
    ):
        super().__init__()

        self.config = config
        self.hc_count = config.hc_count
        if config.mtp_hc and role is not None and "mtp" in role:
            self.hc_count = self.hc_count + 1
        self.hidden_size = config.hidden_size
        self.params_dtype = config.params_dtype

    def mix(self, hyper_input: torch.Tensor):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        mixed_input = hyper_input.view(
            *hyper_input.shape[:-1], self.hc_count, self.hidden_size
        ).mean(dim=-2)
        return mixed_input, hyper_input

    def combine(
        self, block_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        assert residual.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual_reshaped = residual.view(
            *residual.shape[:-1], self.hc_count, self.hidden_size
        )
        combined_output = residual_reshaped + block_output.unsqueeze(-2)
        combined_output = combined_output.view(
            *residual.shape[:-1], self.hc_count * self.hidden_size
        )
        return combined_output


class GatedResidual(HyperConnectionBase):
    # The next read's (normalized, mixed) input, made by the preceding write-back's one-launch boundary.
    _premixed: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    # Largest write-back row count this module may run as a one-launch boundary, in either role; 0 = never.
    _ks_boundary_max_rows: int = 0

    # Import-time selection keeps the disabled entry and launch body stock.
    if _Q38C_EXTEND_PATH:
        def __init__(
            self,
            config: HyperConnectionConfig,
            use_mix: bool = True,
            use_combine: bool = True,
            role: Optional[str] = None,
        ):
            super().__init__(config, use_mix, use_combine, role)

            norm_dim = (
                self.config.hidden_size * self.hc_count
                if self.config.hc_per_branch_norm
                else self.config.hidden_size
            )
            norm_group_size = (
                self.config.hidden_size if self.config.hc_per_branch_norm else None
            )
            self.hc_norm = GroupedGemmaRMSNorm(
                norm_dim, eps=self.config.rms_norm_eps, group_size=norm_group_size
            )

            if use_mix:
                self.input_mix_weight_down = nn.Linear(
                    self.hidden_size * self.hc_count,
                    self.config.hc_lowrank,
                    bias=False,
                    dtype=config.params_dtype,
                )
                self.input_mix_weight_up = nn.Linear(
                    self.config.hc_lowrank,
                    self.hc_count * self.hidden_size,
                    bias=False,
                    dtype=config.params_dtype,
                )
                lowrank = self.config.hc_lowrank
                self._jit_mix_ok = (
                    torch.cuda.is_available()
                    # The CuTe split-K pair is tcgen05 (sm_100 family) only.
                    and torch.cuda.get_device_capability()[0] == 10
                    and (self.hc_count * self.hidden_size) % 2048 == 0
                    and self.hidden_size % 8 == 0
                    and lowrank > 0
                    and lowrank % 8 == 0
                )
                self._mix_up_weight_padded = None

            self._gate_stream = None
            self._splitk_mix_max_rows = (
                _SPLITK_MIX_MAX_ROWS if get_parallel().tp_size == 1 else 24
            )
            # Measured at TP1 with hc 4 only; 0 keeps the torch.compile chain above the 32-row pair.
            self._splitk_mix_wide_max_rows = (
                _SPLITK_MIX_WIDE_MAX_ROWS
                if get_parallel().tp_size == 1
                and self.hc_count == 4
                and _SPLITK_MIX_WIDE_MAX_ROWS > self._splitk_mix_max_rows
                else 0
            )
            self._macro_ok = (
                _HC_MACRO
                and get_parallel().tp_size == 1
                and self.hc_count == 4
                and self.hidden_size == 2560
                and config.hc_per_branch_norm
                and config.params_dtype == torch.bfloat16
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] == 10
            )
            self._fused_combine_ok = (
                _FUSED_COMBINE_MAX_ROWS >= _FUSED_COMBINE_MIN_ROWS
                and get_parallel().tp_size == 1
                and self.hc_count == 4
                and self.hidden_size == 2560
                and config.hc_per_branch_norm
                and config.params_dtype == torch.bfloat16
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] == 10
            )
            self._macro_gate_ok = False
            # The existing BF16 combine/next-norm kernel admits at most four rows.
            # Resolve once before graph capture; the default retains the M=1 path.
            self._gate_max_rows = (
                16 if envs.SGLANG_Q38_HC_SMALLM16.get()
                else 4 if envs.SGLANG_Q38_HC_SMALLM4.get() else 1
            )
            if os.environ.get("SGLANG_Q38C_HC_P2_M16_GATE", "0") == "1":
                self._gate_max_rows = 32
            self._hc_combine_parts = (
                envs.SGLANG_Q38_HC_COMBINE_PARTS.get()
                if envs.SGLANG_Q38_HC_SMALLM16.get() else 2
            )
            if self._hc_combine_parts not in (1, 2):
                raise ValueError("HC combine parts must be 1 or 2")
            self._ext_ok = False
            if use_combine:
                self.block_inject_weight = nn.Linear(
                    self.hidden_size * self.hc_count,
                    self.hc_count,
                    bias=False,
                    dtype=config.params_dtype,
                )
                # hc_combine rejects other shapes; device and dtype are checked per call.
                self._jit_combine_ok = (
                    self.hidden_size % 8 == 0
                    and (self.hc_count * self.hidden_size) % 2048 == 0
                )
                vecs = self.hc_count * self.hidden_size // 8
                self._split_combine_ok = (
                    self._jit_combine_ok
                    and vecs % (8 * 160) == 0
                    and (self.hidden_size // 8) % (vecs // 8) == 0
                )
                # The macro gate runs on the current stream, before the combine; the
                # stock one-row gate keeps its side stream unless the macro takes
                # one row too.
                self._macro_gate_ok = self._macro_ok
                if (not self._macro_ok or _HC_MACRO_MIN_ROWS > 1 or self._fused_combine_ok) and (
                    self.hc_count == 4
                    and self.hidden_size == 2560
                    and config.hc_per_branch_norm
                    and config.params_dtype == torch.bfloat16
                    and torch.cuda.is_available()
                    and torch.cuda.get_device_capability()[0] == 10
                ):
                    # Create before graph capture; all layers on a device share it.
                    self._gate_stream = _hc_gate_stream(torch.cuda.current_device())
                # The extend boundary replays grouped_gemma_rmsnorm's Blackwell
                # (32-byte vector) reduction order; its source refuses to build
                # for other targets, which leaves the stock path in place.
                self._ext_ok = (
                    _HC_EXT
                    and config.hc_per_branch_norm
                    and self.hc_norm._jit_group_size == self.hidden_size
                    and self.hidden_size % 512 == 0
                    and (self.hc_count * self.hidden_size) % 2048 == 0
                    and self.hc_count <= 8
                    and config.params_dtype in (torch.bfloat16, torch.float16)
                    and torch.cuda.is_available()
                    and torch.cuda.get_device_capability()[0] >= 10
                )
                # Build (or load from the JIT cache) now, not in the first
                # extend's serving path; a failed build keeps the stock boundary.
                self._ext_ok = self._ext_ok and _hc_ext_ready(
                    self.hc_count, self.hidden_size, config.params_dtype
                )
            # As a write-back it needs the extend boundary; as the next read, the mix weights.
            if (
                _KS_BOUNDARY_MAX_ROWS > _HC_MACRO_MAX_ROWS
                and self._macro_ok
                and use_mix
                and self._jit_mix_ok
                and config.hc_lowrank == 320
                and (self._ext_ok or not use_combine)
                and _hc_boundary_ks_ready()
            ):
                self._ks_boundary_max_rows = _KS_BOUNDARY_MAX_ROWS

            def _mix_compute(
                hyper_input_normed: torch.Tensor,
                input_mix_weight_down: torch.Tensor,
                input_mix_weight_up: torch.Tensor,
                hc: int,
                hs: int,
            ) -> torch.Tensor:
                input_mix_weight = F.silu(
                    F.linear(hyper_input_normed, input_mix_weight_down) / hc
                )
                input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
                input_mix_weight = torch.sigmoid(input_mix_weight)
                input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
                output = (
                    input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
                ).mean(dim=-2)
                return output

            def _combine_compute(
                block_output: torch.Tensor,
                residual: torch.Tensor,
                normed_residual: torch.Tensor,
                block_inject_weight: torch.Tensor,
                hc: int,
                hs: int,
            ) -> torch.Tensor:
                R = residual.unflatten(-1, (hc, hs))
                block_inject_weight_out = 2 * torch.sigmoid(
                    F.linear(normed_residual, block_inject_weight) / hc
                )
                injection = block_output.unsqueeze(-2) * block_inject_weight_out.unsqueeze(
                    -1
                )
                return (R + injection).flatten(-2)

            self._mix_compute = torch.compile(_mix_compute)
            self._combine_compute = torch.compile(_combine_compute)
    else:
        def __init__(
            self,
            config: HyperConnectionConfig,
            use_mix: bool = True,
            use_combine: bool = True,
            role: Optional[str] = None,
        ):
            super().__init__(config, use_mix, use_combine, role)

            norm_dim = (
                self.config.hidden_size * self.hc_count
                if self.config.hc_per_branch_norm
                else self.config.hidden_size
            )
            norm_group_size = (
                self.config.hidden_size if self.config.hc_per_branch_norm else None
            )
            self.hc_norm = GroupedGemmaRMSNorm(
                norm_dim, eps=self.config.rms_norm_eps, group_size=norm_group_size
            )

            if use_mix:
                self.input_mix_weight_down = nn.Linear(
                    self.hidden_size * self.hc_count,
                    self.config.hc_lowrank,
                    bias=False,
                    dtype=config.params_dtype,
                )
                self.input_mix_weight_up = nn.Linear(
                    self.config.hc_lowrank,
                    self.hc_count * self.hidden_size,
                    bias=False,
                    dtype=config.params_dtype,
                )
                lowrank = self.config.hc_lowrank
                self._jit_mix_ok = (
                    torch.cuda.is_available()
                    # The CuTe split-K pair is tcgen05 (sm_100 family) only.
                    and torch.cuda.get_device_capability()[0] == 10
                    and (self.hc_count * self.hidden_size) % 2048 == 0
                    and self.hidden_size % 8 == 0
                    and lowrank > 0
                    and lowrank % 8 == 0
                )
                self._mix_up_weight_padded = None

            self._gate_stream = None
            self._splitk_mix_max_rows = (
                _SPLITK_MIX_MAX_ROWS if get_parallel().tp_size == 1 else 24
            )
            # Measured at TP1 with hc 4 only; 0 keeps the torch.compile chain above the 32-row pair.
            self._splitk_mix_wide_max_rows = (
                _SPLITK_MIX_WIDE_MAX_ROWS
                if get_parallel().tp_size == 1
                and self.hc_count == 4
                and _SPLITK_MIX_WIDE_MAX_ROWS > self._splitk_mix_max_rows
                else 0
            )
            self._macro_ok = (
                _HC_MACRO
                and get_parallel().tp_size == 1
                and self.hc_count == 4
                and self.hidden_size == 2560
                and config.hc_per_branch_norm
                and config.params_dtype == torch.bfloat16
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] == 10
            )
            self._fused_combine_ok = (
                _FUSED_COMBINE_MAX_ROWS >= _FUSED_COMBINE_MIN_ROWS
                and get_parallel().tp_size == 1
                and self.hc_count == 4
                and self.hidden_size == 2560
                and config.hc_per_branch_norm
                and config.params_dtype == torch.bfloat16
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] == 10
            )
            self._macro_gate_ok = False
            # The existing BF16 combine/next-norm kernel admits at most four rows.
            # Resolve once before graph capture; the default retains the M=1 path.
            self._gate_max_rows = (
                16 if envs.SGLANG_Q38_HC_SMALLM16.get()
                else 4 if envs.SGLANG_Q38_HC_SMALLM4.get() else 1
            )
            if os.environ.get("SGLANG_Q38C_HC_P2_M16_GATE", "0") == "1":
                self._gate_max_rows = 32
            self._hc_combine_parts = (
                envs.SGLANG_Q38_HC_COMBINE_PARTS.get()
                if envs.SGLANG_Q38_HC_SMALLM16.get() else 2
            )
            if self._hc_combine_parts not in (1, 2):
                raise ValueError("HC combine parts must be 1 or 2")
            if use_combine:
                self.block_inject_weight = nn.Linear(
                    self.hidden_size * self.hc_count,
                    self.hc_count,
                    bias=False,
                    dtype=config.params_dtype,
                )
                # hc_combine rejects other shapes; device and dtype are checked per call.
                self._jit_combine_ok = (
                    self.hidden_size % 8 == 0
                    and (self.hc_count * self.hidden_size) % 2048 == 0
                )
                vecs = self.hc_count * self.hidden_size // 8
                self._split_combine_ok = (
                    self._jit_combine_ok
                    and vecs % (8 * 160) == 0
                    and (self.hidden_size // 8) % (vecs // 8) == 0
                )
                # The macro gate runs on the current stream, before the combine; the
                # stock one-row gate keeps its side stream unless the macro takes
                # one row too.
                self._macro_gate_ok = self._macro_ok
                if (not self._macro_ok or _HC_MACRO_MIN_ROWS > 1 or self._fused_combine_ok) and (
                    self.hc_count == 4
                    and self.hidden_size == 2560
                    and config.hc_per_branch_norm
                    and config.params_dtype == torch.bfloat16
                    and torch.cuda.is_available()
                    and torch.cuda.get_device_capability()[0] == 10
                ):
                    # Create before graph capture; all layers on a device share it.
                    self._gate_stream = _hc_gate_stream(torch.cuda.current_device())

            def _mix_compute(
                hyper_input_normed: torch.Tensor,
                input_mix_weight_down: torch.Tensor,
                input_mix_weight_up: torch.Tensor,
                hc: int,
                hs: int,
            ) -> torch.Tensor:
                input_mix_weight = F.silu(
                    F.linear(hyper_input_normed, input_mix_weight_down) / hc
                )
                input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
                input_mix_weight = torch.sigmoid(input_mix_weight)
                input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
                output = (
                    input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
                ).mean(dim=-2)
                return output

            def _combine_compute(
                block_output: torch.Tensor,
                residual: torch.Tensor,
                normed_residual: torch.Tensor,
                block_inject_weight: torch.Tensor,
                hc: int,
                hs: int,
            ) -> torch.Tensor:
                R = residual.unflatten(-1, (hc, hs))
                block_inject_weight_out = 2 * torch.sigmoid(
                    F.linear(normed_residual, block_inject_weight) / hc
                )
                injection = block_output.unsqueeze(-2) * block_inject_weight_out.unsqueeze(
                    -1
                )
                return (R + injection).flatten(-2)

            self._mix_compute = torch.compile(_mix_compute)
            self._combine_compute = torch.compile(_combine_compute)

    def _fused_combine_rows(self, rows: int) -> bool:
        return (
            self._fused_combine_ok
            and _FUSED_COMBINE_MIN_ROWS <= rows <= _FUSED_COMBINE_MAX_ROWS
        )

    def _macro_rows(self, rows: int) -> bool:
        """Whether the fused boundary serves this many rows; elsewhere the gate
        partials, if any, come from the stock side stream."""
        return self._macro_ok and _HC_MACRO_MIN_ROWS <= rows <= _HC_MACRO_MAX_ROWS

    def all_reduce_combine_and_normalize(
        self,
        block_partial: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        next_norm: Optional[GroupedGemmaRMSNorm],
        coordinator,
        site: str,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, None]]:
        """``combine_and_normalize(coordinator.all_reduce(block_partial), ...)``
        where that takes the side-stream-gate ``hc_combine_apply_norm`` branch and
        the all-reduce is the CustomAllReduceV2 1shot push: one kernel, the same
        bits. None, with nothing launched, anywhere else."""
        from sglang.kernels.ops.communication import hc_all_reduce_combine_norm as fused

        hyper_input, _, partials = residuals
        rows = block_partial.shape[0]
        if not (
            _HC_AR_COMBINE_NORM
            and self.hc_count == fused.HC_COUNT
            and self.hidden_size == fused.HIDDEN
            and 0 < rows <= fused.MAX_ROWS
            # the stock decode branch: side-stream partials, no macro, a per-branch next norm
            and partials is not None
            and partials.dim() == 3
            and tuple(partials.shape) == (rows, 8, self.hc_count)
            and partials.dtype == torch.float32
            and self._gate_stream is not None
            and not self._macro_rows(rows)
            and next_norm is not None
            and next_norm.group_size == self.hidden_size
            and next_norm.weight.dtype == torch.bfloat16
            and block_partial.dtype == torch.bfloat16
            and hyper_input.dtype == torch.bfloat16
            and tuple(block_partial.shape) == (rows, self.hidden_size)
            and tuple(hyper_input.shape) == (rows, self.hc_count * self.hidden_size)
            and block_partial.is_contiguous()
            and hyper_input.is_contiguous()
            and partials.is_contiguous()
            # the kernel's 32B vector accesses
            and block_partial.data_ptr() % 32 == 0
            and hyper_input.data_ptr() % 32 == 0
            and next_norm.weight.data_ptr() % 32 == 0
            and not torch.compiler.is_compiling()
        ):
            return None
        plane = fused.one_shot_push_plane(coordinator, block_partial)
        if plane is None:
            return None
        current = torch.cuda.current_stream()
        current.wait_stream(self._gate_stream)
        partials.record_stream(current)
        combined, normalized = fused.hc_all_reduce_combine_norm(
            plane.obj,
            block_partial,
            hyper_input,
            partials,
            next_norm.weight,
            next_norm.variance_epsilon,
        )
        if rows not in _HC_AR_LOGGED or _HC_AR_VERBOSE:
            _HC_AR_LOGGED.add(rows)
            logger.info(
                "SGLANG_HC_AR_COMBINE_NORM: TP all-reduce fused with the HC "
                "combine and next norm, site=%s rows=%d group=%s world=%d slot=%dB",
                site,
                rows,
                coordinator.unique_name,
                coordinator.world_size,
                plane.max_push_size,
            )
        return combined, normalized, None

    def _ext_rows(
        self,
        block_output: torch.Tensor,
        hyper_input: torch.Tensor,
        hyper_input_normed: torch.Tensor,
        partials: Optional[torch.Tensor],
    ) -> bool:
        """Whether the extend boundary serves this write-back. Its partials
        are [rows, hc] fp32 gate dot totals, made only by itself."""
        rows = block_output.shape[0]
        if not (self._ext_ok and rows >= _HC_EXT_MIN_ROWS):
            return False
        dtype = block_output.dtype
        if partials is None:
            partials_ok = (
                hyper_input_normed.dtype == dtype
                and hyper_input_normed.shape == hyper_input.shape
            )
        else:
            partials_ok = partials.dtype == torch.float32 and tuple(
                partials.shape
            ) == (rows, self.hc_count)
        return (
            partials_ok
            and block_output.dim() == 2
            and hyper_input.dim() == 2
            and hyper_input.dtype == dtype
            and self.block_inject_weight.weight.dtype == dtype
            and block_output.is_cuda
            and not torch.compiler.is_compiling()
        )

    def _combine_extend(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        next_norm: Optional[GroupedGemmaRMSNorm],
        next_inject_weight: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        from sglang.kernels.ops.elementwise.hc_combine_extend import (
            hc_combine_extend,
        )

        hyper_input, hyper_input_normed, partials = residuals
        dtype = block_output.dtype
        width = self.hc_count * self.hidden_size
        if next_norm is not None and not (
            next_norm.group_size == self.hidden_size
            and next_norm._jit_group_size == self.hidden_size
            and next_norm.weight.dtype == dtype
            and next_norm.weight.numel() == width
        ):
            next_norm = None
        if next_norm is None or (
            next_inject_weight is not None
            and (
                next_inject_weight.dtype != dtype
                or tuple(next_inject_weight.shape) != (self.hc_count, width)
            )
        ):
            next_inject_weight = None
        _log_hc_ext_once(
            block_output.shape[0],
            partials is not None,
            next_norm is not None,
            next_inject_weight is not None,
        )
        return hc_combine_extend(
            block_output,
            hyper_input,
            self.hc_count,
            self.hidden_size,
            partials=partials,
            normed_residual=hyper_input_normed if partials is None else None,
            inject_weight=(
                self.block_inject_weight.weight if partials is None else None
            ),
            norm_weight=next_norm.weight if next_norm is not None else None,
            eps=next_norm.variance_epsilon if next_norm is not None else 0.0,
            next_inject_weight=next_inject_weight,
        )

    def _ks_boundary_admits(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        next_norm: Optional[GroupedGemmaRMSNorm],
        next_inject_weight: Optional[torch.Tensor],
        next_mix: "GatedResidual",
    ) -> bool:
        """Whether this write-back and next_mix's read run as one launch: the extend boundary with its [rows, hc]
        gate totals, next_mix's own per-branch norm and gate, contiguous rows, a resident grid."""
        rows = block_output.shape[0]
        hyper_input, hyper_input_normed, partials = residuals
        if not (
            _HC_MACRO_MAX_ROWS < rows <= min(self._ks_boundary_max_rows, next_mix._ks_boundary_max_rows)
            and partials is not None
            and partials.dim() == 2
            and next_norm is next_mix.hc_norm
            and next_norm._jit_group_size == self.hidden_size
            and next_norm.weight.dtype == block_output.dtype
            and (
                next_inject_weight is None
                or next_inject_weight is next_mix.block_inject_weight.weight
            )
            and self._ext_rows(block_output, hyper_input, hyper_input_normed, partials)
            and block_output.is_contiguous()
            and hyper_input.is_contiguous()
            and partials.is_contiguous()
        ):
            return False
        from sglang.kernels.ops.gemm.hc_boundary_ks import supported

        return supported(rows, block_output.device.index)

    def _combine_mix_ks(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        next_norm: GroupedGemmaRMSNorm,
        next_inject_weight: Optional[torch.Tensor],
        next_mix: "GatedResidual",
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """The write-back and next_mix's read in one launch; next_mix.mix() returns the stashed mixed input."""
        from sglang.kernels.ops.gemm.hc_boundary_ks import FLAG_WRITE_NORMED, hc_boundary_ks
        from sglang.kernels.ops.gemm.hc_mix_splitk import permute_pad_up_weight

        hyper_input, _, partials = residuals
        if next_mix._mix_up_weight_padded is None:
            next_mix._mix_up_weight_padded = permute_pad_up_weight(
                next_mix.input_mix_weight_up.weight, next_mix.hc_count
            )
        # With next gate totals and next_mix's extend boundary, next_mix's write-back reads the totals, never the
        # normalized streams: they are not written, and a zero-size stand-in carries the hand-off (any other
        # reader fails mix()'s shape check).
        write_normed = bool(_KS_BOUNDARY_FLAGS & FLAG_WRITE_NORMED) or not (
            next_inject_weight is not None and next_mix._ext_ok
        )
        combined, normalized, next_partials, mixed = hc_boundary_ks(
            block_output,
            hyper_input,
            partials=partials,
            norm_weight=next_norm.weight,
            eps=next_norm.variance_epsilon,
            next_inject_weight=next_inject_weight,
            w_down=next_mix.input_mix_weight_down.weight.data,
            w_up_permuted=next_mix._mix_up_weight_padded,
            write_normed=write_normed,
            flags=_KS_BOUNDARY_FLAGS,
        )
        if normalized is None:
            normalized = hyper_input.new_empty((0,))
        next_mix._premixed = (normalized, mixed)
        return combined, normalized, next_partials

    def mix(
        self,
        hyper_input: torch.Tensor,
        normalized_input: Optional[torch.Tensor] = None,
        partials: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Mix the streams down for a block; return it and the write-back state.

        ``partials`` are this read's gate partials, prepared with
        ``normalized_input`` by the preceding ``combine_and_normalize``.
        """
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed_input = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype
            )
            return mixed_input, (hyper_input, hyper_input, None)
        premixed = self._premixed
        if premixed is not None:
            self._premixed = None
            if normalized_input is premixed[0]:
                return premixed[1], (hyper_input, normalized_input, partials)

        if partials is not None:
            assert normalized_input is not None
        if normalized_input is not None:
            assert self.config.hc_per_branch_norm
            assert normalized_input.shape == hyper_input.shape
            hyper_input_normed = normalized_input
        elif self.config.hc_per_branch_norm:
            hyper_input_normed = self.hc_norm(hyper_input)
        else:
            hyper_input_normed = self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        # First/PLE/MTP-entry reads have no prepared gate from a predecessor.
        if (
            partials is None
            and self._macro_gate_ok
            and self._macro_rows(hyper_input.shape[0])
            and hyper_input_normed.dtype == torch.bfloat16
            and not torch.compiler.is_compiling()
        ):
            from sglang.kernels.ops.elementwise.hc_combine_decode import hc_combine_gate

            partials = hc_combine_gate(
                hyper_input_normed, self.block_inject_weight.weight
            )
        # Retain FN's side-stream producer for all non-macro row counts.
        stream = self._gate_stream
        fused_combine_rows = (
            self._fused_combine_rows(hyper_input.shape[0])
            and hyper_input_normed.dtype == torch.bfloat16
            and not torch.compiler.is_compiling()
        )
        if (
            stream is not None
            and partials is None
            and not self._macro_rows(hyper_input.shape[0])
            and (0 < hyper_input.shape[0] <= self._gate_max_rows or fused_combine_rows)
            and hyper_input.dtype == torch.bfloat16
            and not torch.compiler.is_compiling()
        ):
            from sglang.kernels.ops.elementwise.hc_combine_decode import hc_combine_gate

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                partials = hc_combine_gate(
                    hyper_input_normed, self.block_inject_weight.weight
                )
            hyper_input_normed.record_stream(stream)
        if (
            self._jit_mix_ok
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and hyper_input_normed.shape[0] <= self._splitk_mix_max_rows
        ):
            from sglang.kernels.ops.gemm.hc_mix_splitk import (
                hc_mix,
                permute_pad_up_weight,
            )

            if self._mix_up_weight_padded is None:
                self._mix_up_weight_padded = permute_pad_up_weight(
                    self.input_mix_weight_up.weight, self.hc_count
                )
            mixed_input = hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight.data,
                self._mix_up_weight_padded,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif (
            self._jit_mix_ok
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and hyper_input_normed.shape[0] <= self._splitk_mix_wide_max_rows
            and hyper_input_normed.is_contiguous()
            and not torch.compiler.is_compiling()
        ):
            from sglang.kernels.ops.gemm.hc_mix_splitk import (
                hc_mix_wide,
                permute_pad_up_weight,
            )

            if self._mix_up_weight_padded is None:
                self._mix_up_weight_padded = permute_pad_up_weight(
                    self.input_mix_weight_up.weight, self.hc_count
                )
            mixed_input = hc_mix_wide(
                hyper_input_normed,
                self.input_mix_weight_down.weight.data,
                self._mix_up_weight_padded,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif fused_hc_mix_supported(
            hyper_input_normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        ):
            mixed_input = fused_hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        else:
            mix_compute = (
                _pinned_mix_compute()
                if _mix_gemms_pinned(self.config.hc_lowrank, self.hc_count * self.hidden_size)
                else self._mix_compute
            )
            mixed_input = mix_compute(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        return mixed_input, (hyper_input, hyper_input_normed, partials)

    # Import-time selection keeps the disabled entry and launch body stock.
    if _Q38C_EXTEND_PATH:
        def combine_and_normalize(
            self,
            block_output: torch.Tensor,
            residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            next_norm: Optional[GroupedGemmaRMSNorm],
            next_inject_weight: Optional[torch.Tensor] = None,
            next_mix: Optional["GatedResidual"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            """Return the residual, its next HC normalization and the next gate
            partials explicitly: ``(combined, normalized, partials)``.

            The BF16 combine result is retained before calculating the next norm.
            Callers must only pass the norm when no PLE or other update intervenes,
            and ``next_inject_weight`` only with the norm of the read it gates.
            The next normalization and partials are None where not prepared.
            """
            hyper_input, hyper_input_normed, partials = residuals
            if (
                next_mix is not None
                and self._ks_boundary_max_rows
                and next_norm is not None
                and self._ks_boundary_admits(
                    block_output, residuals, next_norm, next_inject_weight, next_mix
                )
            ):
                return self._combine_mix_ks(
                    block_output, residuals, next_norm, next_inject_weight, next_mix
                )
            if self._ext_rows(block_output, hyper_input, hyper_input_normed, partials):
                return self._combine_extend(
                    block_output, residuals, next_norm, next_inject_weight
                )
            if partials is not None and partials.dim() == 2:
                # The extend boundary's [rows, hc] gate totals, which the stock
                # kernels (decode [rows, 8, hc] partials) cannot read; the gate
                # recomputed from the normalized streams is the same.
                partials = None
                residuals = (hyper_input, hyper_input_normed, None)
            if (
                partials is not None
                and self._macro_rows(block_output.shape[0])
                and next_norm is not None
                and next_norm.group_size == self.hidden_size
                and next_norm.weight.dtype == torch.bfloat16
                and block_output.dtype == torch.bfloat16
                and hyper_input.dtype == torch.bfloat16
                and not torch.compiler.is_compiling()
            ):
                from sglang.kernels.ops.elementwise.hc_combine_decode import (
                    hc_combine_norm_gate,
                )

                if next_inject_weight is not None and (
                    next_inject_weight.dtype != torch.bfloat16
                    or tuple(next_inject_weight.shape)
                    != (self.hc_count, self.hc_count * self.hidden_size)
                ):
                    next_inject_weight = None
                return hc_combine_norm_gate(
                    block_output,
                    hyper_input,
                    partials,
                    next_norm.weight,
                    next_norm.variance_epsilon,
                    next_inject_weight,
                )
            if (
                partials is not None
                # Stock: only the side-stream gate (one row) prepares partials.
                and self._gate_stream is not None
                and not self._macro_rows(block_output.shape[0])
                and next_norm is not None
                and next_norm.group_size == self.hidden_size
                and next_norm.weight.dtype == torch.bfloat16
                and block_output.dtype == torch.bfloat16
            ):
                from sglang.kernels.ops.elementwise.hc_combine_decode import (
                    hc_combine_apply_norm,
                )

                current = torch.cuda.current_stream()
                current.wait_stream(self._gate_stream)
                partials.record_stream(current)
                combined, normalized = hc_combine_apply_norm(
                    block_output,
                    hyper_input,
                    partials,
                    next_norm.weight,
                    next_norm.variance_epsilon,
                    parts=self._hc_combine_parts,
                )
                return combined, normalized, None
            return self.combine(block_output, residuals), None, None
    else:
        def combine_and_normalize(
            self,
            block_output: torch.Tensor,
            residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            next_norm: Optional[GroupedGemmaRMSNorm],
            next_inject_weight: Optional[torch.Tensor] = None,
            next_mix: Optional["GatedResidual"] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            """Return the residual, its next HC normalization and the next gate
            partials explicitly: ``(combined, normalized, partials)``.

            The BF16 combine result is retained before calculating the next norm.
            Callers must only pass the norm when no PLE or other update intervenes,
            and ``next_inject_weight`` only with the norm of the read it gates.
            The next normalization and partials are None where not prepared.
            """
            hyper_input, _, partials = residuals
            if (
                partials is not None
                and self._macro_rows(block_output.shape[0])
                and next_norm is not None
                and next_norm.group_size == self.hidden_size
                and next_norm.weight.dtype == torch.bfloat16
                and block_output.dtype == torch.bfloat16
                and hyper_input.dtype == torch.bfloat16
                and not torch.compiler.is_compiling()
            ):
                from sglang.kernels.ops.elementwise.hc_combine_decode import (
                    hc_combine_norm_gate,
                )

                if next_inject_weight is not None and (
                    next_inject_weight.dtype != torch.bfloat16
                    or tuple(next_inject_weight.shape)
                    != (self.hc_count, self.hc_count * self.hidden_size)
                ):
                    next_inject_weight = None
                return hc_combine_norm_gate(
                    block_output,
                    hyper_input,
                    partials,
                    next_norm.weight,
                    next_norm.variance_epsilon,
                    next_inject_weight,
                )
            if (
                partials is not None
                # Stock: only the side-stream gate (one row) prepares partials.
                and self._gate_stream is not None
                and not self._macro_rows(block_output.shape[0])
                and next_norm is not None
                and next_norm.group_size == self.hidden_size
                and next_norm.weight.dtype == torch.bfloat16
                and block_output.dtype == torch.bfloat16
            ):
                from sglang.kernels.ops.elementwise.hc_combine_decode import (
                    hc_combine_apply_norm,
                )

                current = torch.cuda.current_stream()
                current.wait_stream(self._gate_stream)
                partials.record_stream(current)
                combined, normalized = hc_combine_apply_norm(
                    block_output,
                    hyper_input,
                    partials,
                    next_norm.weight,
                    next_norm.variance_epsilon,
                    parts=self._hc_combine_parts,
                )
                return combined, normalized, None
            return self.combine(block_output, residuals), None, None

    # Import-time selection keeps the disabled entry and launch body stock.
    if _Q38C_EXTEND_PATH:
        def combine(
            self,
            block_output: torch.Tensor,
            residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        ) -> torch.Tensor:
            hyper_input, hyper_input_normed, partials = residuals
            if self._ext_rows(block_output, hyper_input, hyper_input_normed, partials):
                return self._combine_extend(block_output, residuals, None, None)[0]
            if partials is not None and partials.dim() == 2:
                # Extend-boundary gate totals: recompute from the normalized streams.
                partials = None
            if partials is not None:
                from sglang.kernels.ops.elementwise.hc_combine_decode import (
                    hc_combine_apply,
                )

                if self._gate_stream is not None and not self._macro_rows(
                    block_output.shape[0]
                ):
                    # Made on the side stream; the macro path makes them on this one.
                    current = torch.cuda.current_stream()
                    current.wait_stream(self._gate_stream)
                    partials.record_stream(current)
                return hc_combine_apply(block_output, hyper_input, partials)
            assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
            assert block_output.shape[-1] == self.hidden_size
            if block_output.shape[0] == 0:
                return hyper_input.to(self.params_dtype)

            if (
                self._jit_combine_ok
                and block_output.is_cuda
                and block_output.dtype in (torch.bfloat16, torch.float16)
                and hyper_input.dtype == block_output.dtype
                and hyper_input_normed.dtype == block_output.dtype
                and self.block_inject_weight.weight.dtype == block_output.dtype
            ):
                if self._split_combine_ok and block_output.shape[0] <= 32:
                    from sglang.kernels.ops.elementwise.hc_combine import (
                        hc_combine_split,
                    )

                    return hc_combine_split(
                        block_output,
                        hyper_input,
                        hyper_input_normed,
                        self.block_inject_weight.weight.data,
                        self.hc_count,
                        self.hidden_size,
                    )
                from sglang.kernels.ops.elementwise.hc_combine import hc_combine

                return hc_combine(
                    block_output,
                    hyper_input,
                    hyper_input_normed,
                    self.block_inject_weight.weight,
                    self.hc_count,
                    self.hidden_size,
                )

            updated_residuals = self._combine_compute(
                block_output,
                hyper_input,
                hyper_input_normed,
                self.block_inject_weight.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
            return updated_residuals
    else:
        def combine(
            self,
            block_output: torch.Tensor,
            residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        ) -> torch.Tensor:
            hyper_input, hyper_input_normed, partials = residuals
            if partials is not None:
                from sglang.kernels.ops.elementwise.hc_combine_decode import (
                    hc_combine_apply,
                )

                if self._gate_stream is not None and not self._macro_rows(
                    block_output.shape[0]
                ):
                    # Made on the side stream; the macro path makes them on this one.
                    current = torch.cuda.current_stream()
                    current.wait_stream(self._gate_stream)
                    partials.record_stream(current)
                return hc_combine_apply(block_output, hyper_input, partials)
            assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
            assert block_output.shape[-1] == self.hidden_size
            if block_output.shape[0] == 0:
                return hyper_input.to(self.params_dtype)

            if (
                self._jit_combine_ok
                and block_output.is_cuda
                and block_output.dtype in (torch.bfloat16, torch.float16)
                and hyper_input.dtype == block_output.dtype
                and hyper_input_normed.dtype == block_output.dtype
                and self.block_inject_weight.weight.dtype == block_output.dtype
            ):
                if self._split_combine_ok and block_output.shape[0] <= 32:
                    from sglang.kernels.ops.elementwise.hc_combine import (
                        hc_combine_split,
                    )

                    return hc_combine_split(
                        block_output,
                        hyper_input,
                        hyper_input_normed,
                        self.block_inject_weight.weight.data,
                        self.hc_count,
                        self.hidden_size,
                    )
                from sglang.kernels.ops.elementwise.hc_combine import hc_combine

                return hc_combine(
                    block_output,
                    hyper_input,
                    hyper_input_normed,
                    self.block_inject_weight.weight,
                    self.hc_count,
                    self.hidden_size,
                )

            updated_residuals = self._combine_compute(
                block_output,
                hyper_input,
                hyper_input_normed,
                self.block_inject_weight.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
            return updated_residuals


HYPERCONNECTION_CLASS_DICT = {
    "hyperconnection_average": HyperConnectionBase,
    "gated_residual_simple": GatedResidual,
}
