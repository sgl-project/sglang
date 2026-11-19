"""Tuned cuBLASLt algorithms for the per-tensor FP8 GEMM through FlashInfer's ``bmm_fp8``.

FlashInfer's cuBLAS FP8 runner uses the cuBLASLt heuristic's algorithm, or an
autotuned tactic that indexes the algorithm list cuBLASLt enumerates at the actual
row count, so the kernel that runs at a given (N, K, M) need not be the fastest one.

A table under ``configs/bmm_fp8_cublaslt_algos/`` maps an exact (N, K, M) to a
``cublasLtMatmulAlgo_t`` for one GPU and one cuBLASLt and FlashInfer version. Each
entry is the fastest measured algorithm whose output is byte-equal to the default
algorithm's. Coverage: E4M3 x E4M3 -> BF16, row-major A, column-major B. Any other
call, a shape not in the table, another GPU or library version, FlashInfer internals
other than the ones used here, an algorithm cuBLASLt refuses, or an output that
differs from the stock path's runs the stock path.

Only ``flashinfer_bmm_fp8`` reads the tables. A caller that runs FlashInfer's cuBLAS
FP8 runner itself with an algorithm of its own keeps that algorithm, and a shape such
a caller pins is left out of the tables, so its fallback through
``flashinfer_bmm_fp8`` runs the stock path.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import re
from typing import Callable, Dict, Optional, Tuple

import msgspec
import torch

from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "configs", "bmm_fp8_cublaslt_algos"
)
_ALGO_BYTES = 64  # sizeof(cublasLtMatmulAlgo_t)
_SHAPE_KEY = re.compile(r"N=(\d+),K=(\d+)")


class _Table(msgspec.Struct, frozen=True):
    algos: Dict[Tuple[int, int], Dict[int, torch.Tensor]]  # (N, K) -> M -> algorithm
    run_with_algo: Callable[..., None]  # FlashInfer's bmm_fp8_run_with_algo
    workspace: Callable[[torch.device], torch.Tensor]  # the stock bmm_fp8 workspace


# Per device index; None when no table applies.
_tables: Dict[Optional[int], Optional[_Table]] = {}
_validated: set = set()
_refused: set = set()
_logged: set = set()


def _table_path(device_name: str) -> str:
    name = device_name.replace(" ", "_")
    return os.path.join(_CONFIG_DIR, f"device_name={name}.json")


def _parse_table(doc: dict) -> Dict[Tuple[int, int, int], bytes]:
    table = {}
    for shape, rows in doc["configs"].items():
        match = _SHAPE_KEY.fullmatch(shape)
        if match is None:
            raise ValueError(f"bad shape key {shape!r}")
        n, k = int(match[1]), int(match[2])
        for m, hex_algo in rows.items():
            algo = bytes.fromhex(hex_algo)
            if len(algo) != _ALGO_BYTES:
                raise ValueError(f"{shape} M={m}: {len(algo)} algorithm bytes")
            table[(n, k, int(m))] = algo
    return table


def _device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device)


def _library_versions(cublaslt_major: int) -> Tuple[Optional[int], Optional[str]]:
    try:
        # RTLD_NOLOAD: read the cuBLASLt this process already loaded, never load a copy.
        lib = ctypes.CDLL(f"libcublasLt.so.{cublaslt_major}", mode=os.RTLD_NOLOAD)
        lib.cublasLtGetVersion.restype = ctypes.c_size_t
        cublaslt = int(lib.cublasLtGetVersion())
    except (OSError, AttributeError):
        cublaslt = None
    try:
        import flashinfer

        flashinfer_version = flashinfer.__version__
    except ImportError:
        flashinfer_version = None
    return cublaslt, flashinfer_version


def _flashinfer_entry_points() -> Tuple[Callable[..., None], Callable]:
    """``bmm_fp8_run_with_algo`` of the module FlashInfer's cuBLAS FP8 runner calls, and
    the stock ``bmm_fp8`` workspace getter. Raises if FlashInfer does not expose them."""
    from flashinfer.gemm import gemm_base

    forward = gemm_base.get_gemm_module().cublas_fp8_gemm_runner().forward.__func__
    cells = dict(zip(forward.__code__.co_freevars, forward.__closure__ or ()))
    run_with_algo = cells["module"].cell_contents.bmm_fp8_run_with_algo
    get_buf = gemm_base._get_stream_cache_buf
    size = gemm_base.DEFAULT_WORKSPACE_SIZE

    def workspace(device: torch.device) -> torch.Tensor:
        # the buffer the stock bmm_fp8 passes on the current stream
        return get_buf("bmm_fp8_workspace", size, device)

    return run_with_algo, workspace


def _load(device: torch.device) -> Optional[_Table]:
    name = _device_name(device)
    path = _table_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
        wanted = (doc["meta"]["cublaslt"], doc["meta"]["flashinfer"])
        flat = _parse_table(doc)
        found = _library_versions(wanted[0] // 10000)
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning("bmm_fp8: %s not loaded (%s); default algorithms", path, e)
        return None
    if found != wanted:
        log_info_on_rank0(
            logger,
            f"bmm_fp8: tuned cuBLASLt algorithms for {name} need cuBLASLt "
            f"{wanted[0]} and FlashInfer {wanted[1]}, found {found[0]} and "
            f"{found[1]}; default algorithms",
        )
        return None
    try:
        run_with_algo, workspace = _flashinfer_entry_points()
    except Exception as e:  # FlashInfer internals differ from the ones used here
        log_info_on_rank0(
            logger,
            f"bmm_fp8: FlashInfer {found[1]} does not expose the cuBLAS FP8 "
            f"entry points ({e!r}); default algorithms",
        )
        return None
    algos: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}
    for (n, k, m), algo in flat.items():
        algos.setdefault((n, k), {})[m] = torch.frombuffer(
            bytearray(algo), dtype=torch.uint8
        )
    log_info_on_rank0(
        logger, f"bmm_fp8: {len(flat)} tuned cuBLASLt algorithms for {name}"
    )
    return _Table(algos=algos, run_with_algo=run_with_algo, workspace=workspace)


def _log_once(key: tuple, msg: str) -> None:
    if key not in _logged:
        _logged.add(key)
        log_info_on_rank0(logger, msg)


def _covered(
    q_input: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype
) -> bool:
    return (
        q_input.dtype == torch.float8_e4m3fn
        and weight.dtype == torch.float8_e4m3fn
        and out_dtype == torch.bfloat16
        and q_input.is_contiguous()
        and weight.t().is_contiguous()
    )


def run_with_tuned_algo(
    q_input: torch.Tensor,  # [M, K] e4m3
    weight: torch.Tensor,  # [K, N] e4m3, column-major
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    stock: Callable[[], torch.Tensor],
) -> Optional[torch.Tensor]:
    """The GEMM through the tuned algorithm for this exact shape; None runs ``stock()``.

    The first eager call of a shape also runs ``stock()`` and keeps the algorithm only
    if both outputs are byte-equal; before that, a call inside graph capture gets None.
    """
    device = q_input.device
    if device.index not in _tables:
        _tables[device.index] = _load(device)
    table = _tables[device.index]
    if table is None or not _covered(q_input, weight, out_dtype):
        return None
    m, k = q_input.shape
    n = weight.shape[1]
    rows = table.algos.get((n, k))
    if rows is None:
        return None
    algo = rows.get(m)
    if algo is None:
        _log_once(
            ("miss", n, k),
            f"bmm_fp8: no tuned algorithm for N={n} K={k} at M={m}; default "
            "algorithm (logged once per N, K)",
        )
        return None
    key = (device.index, n, k, m)
    if key in _refused:
        return None
    validated = key in _validated
    if not validated and torch.cuda.is_current_stream_capturing():
        return None
    workspace = table.workspace(device)
    out = torch.empty((1, m, n), dtype=out_dtype, device=device)
    try:
        table.run_with_algo(
            q_input.unsqueeze(0),
            weight.unsqueeze(0),
            out,
            x_scale.reshape(1),
            weight_scale.reshape(1),
            workspace,
            algo,
            0,
        )
    except Exception as e:  # cublasLtMatmul returned an error status for this algorithm
        _refused.add(key)
        logger.warning("bmm_fp8: tuned algorithm for %s refused (%s)", key[1:], e)
        return None
    out = out.view(m, n)
    if not validated:
        ref = stock()
        if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
            _refused.add(key)
            logger.warning(
                "bmm_fp8: tuned algorithm for %s differs from stock", key[1:]
            )
            return ref
        _validated.add(key)
    _log_once(
        ("hit", n, k),
        f"bmm_fp8: tuned cuBLASLt algorithm in use for N={n} K={k} "
        "(logged once per N, K)",
    )
    return out
