"""Pinned cuBLASLt algorithm for the GDN fused in_proj GEMM (qkvz + ba, BF16).

The stock BF16 GEMM dispatch lets cuBLAS pick an algorithm per M. A tuning file
(``SGLANG_FLASHINFER_TACTIC_PIN_FILE``) may instead name, for an M range, the
algorithm cuBLASLt's heuristic returns at another M for the same problem::

    "gdn_in_proj_cublaslt": {"n": 8192, "k": 2048, "m_min": 65, "m_max": 96,
                             "heuristic_m": 128, "algo_index": 1}

The heuristic's list depends on the cuBLAS(Lt) build and the device, so the entry
applies only under the file's exact ``_metadata`` (which names the cuBLAS release,
shared by cuBLASLt, and the SM); see ``flashinfer_tactic_pins``. The GEMM runs
through FlashInfer's mm_bf16_cublaslt module (TRANSA=T, TRANSB=N, FP32 compute,
BF16 output, beta 0). Only an algorithm without split-K keeps the stock reduction
order per output, which is what an entry must be checked for. Any refusal falls
back to the stock call.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import msgspec
import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs
from sglang.srt.layers.moe.flashinfer_tactic_pins import (
    read_tuning_file,
    tuning_file_applies,
)

logger = logging.getLogger(__name__)

_WORKSPACE_BYTES = 32 * 1024 * 1024
# FlashInfer's kMaxAlgorithms and kAlgoBytes (sizeof(cublasLtMatmulAlgo_t)).
_MAX_ALGOS = 100
_ALGO_BYTES = 64
# The call site only offers decode-sized M.
_MAX_M = 1024


class GdnInProjAlgoPin(msgspec.Struct, frozen=True):
    n: int
    k: int
    m_min: int
    m_max: int
    heuristic_m: int
    algo_index: int


def parse_gdn_in_proj_pin(document: dict) -> Optional[GdnInProjAlgoPin]:
    section = document.get("gdn_in_proj_cublaslt")
    if section is None:
        return None
    where = "tactic pin file: gdn_in_proj_cublaslt"
    try:
        pin = GdnInProjAlgoPin(
            **{name: int(section[name]) for name in GdnInProjAlgoPin.__struct_fields__}
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"{where} needs integer {', '.join(GdnInProjAlgoPin.__struct_fields__)}"
        ) from None
    if not (
        pin.n > 0
        and pin.k > 0
        and 1 <= pin.m_min <= pin.m_max <= _MAX_M
        and 1 <= pin.heuristic_m <= _MAX_M
        and 0 <= pin.algo_index < _MAX_ALGOS
    ):
        raise ValueError(f"{where}: out of range: {section!r}")
    return pin


def gdn_in_proj_pin_configured() -> bool:
    """Whether the tuning file has an in_proj entry at all (no device checks)."""
    path = envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.get()
    return bool(path) and parse_gdn_in_proj_pin(read_tuning_file(path)) is not None


@functools.lru_cache(maxsize=1)
def _pin_of(path: str) -> Optional[GdnInProjAlgoPin]:
    pin = parse_gdn_in_proj_pin(read_tuning_file(path))
    if pin is None or not tuning_file_applies(path, need_cache_key_layout=False):
        return None
    return pin


class _Plan(msgspec.Struct):
    module: object
    algo_buf: torch.Tensor
    workspace: torch.Tensor


_plans: dict[tuple, _Plan] = {}
_declined_m: set[int] = set()
_disabled = False


def _plan(weight: torch.Tensor, pin: GdnInProjAlgoPin) -> _Plan:
    """FlashInfer's cuBLASLt module, the heuristic's algorithms at ``heuristic_m``
    and a private workspace, per device."""
    from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

    key = (weight.device.index, pin.heuristic_m, pin.algo_index)
    plan = _plans.get(key)
    if plan is not None:
        return plan
    module = gen_mm_bf16_cublaslt_module().build_and_load()
    n, k = weight.shape
    workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=weight.device)
    algo_buf = torch.zeros(_MAX_ALGOS * _ALGO_BYTES, dtype=torch.uint8, device="cpu")
    probe_a = torch.empty((pin.heuristic_m, k), dtype=torch.bfloat16, device=weight.device)
    probe_out = torch.empty((pin.heuristic_m, n), dtype=torch.bfloat16, device=weight.device)
    with torch.cuda.device(weight.device):
        handle = torch.cuda.current_blas_handle()
    count = int(
        module.mm_bf16_cublaslt_get_algos(
            probe_a, weight, None, probe_out, workspace, handle, algo_buf
        )
    )
    if pin.algo_index >= count:
        raise RuntimeError(
            f"the heuristic at M={pin.heuristic_m} returned {count} algorithms, "
            f"no index {pin.algo_index}"
        )
    plan = _Plan(module=module, algo_buf=algo_buf, workspace=workspace)
    _plans[key] = plan
    start = pin.algo_index * _ALGO_BYTES
    logger.info(
        "GDN in_proj cuBLASLt algorithm pinned: M %d-%d, N=%d K=%d, heuristic M=%d "
        "index %d of %d, algo=%s",
        pin.m_min,
        pin.m_max,
        n,
        k,
        pin.heuristic_m,
        pin.algo_index,
        count,
        bytes(algo_buf[start : start + _ALGO_BYTES].tolist()).hex(),
    )
    return plan


def gdn_in_proj_pinned_gemm(
    x: torch.Tensor, weight: torch.Tensor
) -> Optional[torch.Tensor]:
    """``x @ weight.T`` with the pinned algorithm, or None to run the stock GEMM."""
    global _disabled
    path = envs.SGLANG_FLASHINFER_TACTIC_PIN_FILE.get()
    if _disabled or not path:
        return None
    pin = _pin_of(path)
    if pin is None:
        return None
    m = x.shape[0]
    if not pin.m_min <= m <= pin.m_max or m in _declined_m:
        return None
    if (
        x.ndim != 2
        or tuple(weight.shape) != (pin.n, pin.k)
        or x.shape[1] != pin.k
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or torch.compiler.is_compiling()
        or is_batch_invariant_mode_enabled()
    ):
        return None
    try:
        plan = _plan(weight, pin)
    except Exception as e:  # noqa: BLE001 - no module or algorithm: stock for good
        _disabled = True
        logger.warning("GDN in_proj algorithm pin off, stock GEMM kept: %r", e)
        return None
    try:
        out = torch.empty((m, pin.n), dtype=torch.bfloat16, device=x.device)
        with torch.cuda.device(x.device):
            handle = torch.cuda.current_blas_handle()
        plan.module.mm_bf16_cublaslt_run_with_algo(
            x, weight, None, out, plan.workspace, handle, plan.algo_buf, pin.algo_index
        )
    except Exception as e:  # noqa: BLE001 - refused before enqueueing anything
        _declined_m.add(m)
        logger.warning("GDN in_proj algorithm pin declined M=%d, stock GEMM kept: %r", m, e)
        return None
    return out
