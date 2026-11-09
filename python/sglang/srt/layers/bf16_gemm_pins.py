"""Pinned kernels for unquantized BF16 GEMMs at listed (M, N, K) (default off).

``SGLANG_BF16_GEMM_PIN_FILE`` names a JSON file (``.json.py`` so a source overlay can carry it)::

    {"_metadata": {"flashinfer_version": "0.7.2", "cuda_version": "13.4",
                   "cublas_version": "13.8.1", "gpu": "NVIDIA VR NVL72", "sm": "10.7"},
     "bf16_gemm_pins": [
        {"n": 16480, "k": 2560, "m": 256, "kind": "cublaslt", "algo": "<128 hex>", "workspace": 0},
        {"n": 640, "k": 2560, "m": 256, "kind": "tgv", "tactic": 21},
     ]}

Each entry names, for one exact M (a CUDA-graph batch size), a kernel measured faster than the stock dispatch and
bit-equal to it:

* ``"cublaslt"``: a serialized ``cublasLtMatmulAlgo_t`` (opaque, valid for the cuBLAS build the metadata names) run
  through FlashInfer's mm_bf16_cublaslt module (TRANSA=T, TRANSB=N, FP32 compute, BF16 output, beta 0), as
  ``gdn_in_proj_algo_pin`` does. A pin keeps stock's K reduction order: an unsplit algorithm where stock does not
  split K, the same slicing and reduction where it does.
* ``"tgv"``: sglang's CuTe DSL TGV kernel at a fixed tactic (it never splits K).

The file applies only under its exact ``_metadata`` (``flashinfer_tactic_pins.tuning_file_applies``). Before a key
serves, its first eager call checks it once (seeded random activations, the layer's weight) against the stock
dispatch (``stock``): a pin that fails to launch or differs in a single bit is dropped for good and the stock
dispatch kept. A first call under CUDA graph capture cannot check, so that call keeps the stock dispatch. Any other
shape or M, a bias or addend, a non-contiguous or non-BF16 operand, batch-invariant mode or torch.compile tracing
takes the stock dispatch.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SECTION = "bf16_gemm_pins"
_ALGO_BYTES = 64
_KINDS = ("cublaslt", "tgv")
_CHECK_SEED = 0x4442

_state: Optional[dict] = None  # None: not read yet; {} when off
_runners: dict = {}  # (device, m, n, k) -> callable(x, w) or None (stock)
_lt_module = None
_workspaces: dict = {}


def _parse(document: dict, path: str) -> dict:
    section = document.get(_SECTION)
    if not isinstance(section, list):
        raise ValueError(f"{path}: needs a list {_SECTION!r}")
    entries = {}
    for i, entry in enumerate(section):
        where = f"{path}: {_SECTION}[{i}]"
        try:
            m, n, k = int(entry["m"]), int(entry["n"]), int(entry["k"])
            kind = entry["kind"]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{where} needs integer m, n, k and a kind") from None
        if kind not in _KINDS or min(m, n, k) <= 0:
            raise ValueError(f"{where}: kind must be one of {_KINDS}, m, n, k positive")
        if kind == "cublaslt":
            algo = bytes.fromhex(entry["algo"])
            if len(algo) != _ALGO_BYTES:
                raise ValueError(f"{where}: algo must be {_ALGO_BYTES} bytes")
            spec = ("cublaslt", algo, int(entry.get("workspace", 32 << 20)))
        else:
            spec = ("tgv", int(entry["tactic"]))
        if (m, n, k) in entries:
            raise ValueError(f"{where}: duplicate key {(m, n, k)}")
        entries[(m, n, k)] = spec
    return entries


def _load() -> dict:
    global _state
    if _state is not None:
        return _state
    _state = {}
    path = envs.SGLANG_BF16_GEMM_PIN_FILE.get()
    if not path:
        return _state
    from sglang.srt.layers.moe.flashinfer_tactic_pins import (
        read_tuning_file,
        tuning_file_applies,
    )

    entries = _parse(read_tuning_file(path), path)
    if not tuning_file_applies(path, need_cache_key_layout=False):
        logger.warning("BF16 GEMM pins off: %s is for another stack (see above)", path)
        return _state
    _state = {
        "path": path,
        "entries": entries,
        "shapes": {(n, k) for (_, n, k) in entries},
    }
    logger.info(
        "BF16 GEMM pins on: %d keys over %d shapes from %s",
        len(entries),
        len(_state["shapes"]),
        os.path.basename(path),
    )
    return _state


def enabled() -> bool:
    return bool(_load())


def has_shape(n: int, k: int) -> bool:
    state = _load()
    return bool(state) and (n, k) in state["shapes"]


def _cublaslt_runner(device: torch.device, n: int, k: int, algo: bytes, workspace: int):
    global _lt_module
    if _lt_module is None:
        from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

        _lt_module = gen_mm_bf16_cublaslt_module().build_and_load()
    module = _lt_module
    key = (device.index, n, k)
    ws = _workspaces.get(key)
    if ws is None or ws.numel() < max(workspace, 1):
        # One per shape: two GEMMs of one shape never run concurrently on two streams here.
        ws = torch.empty(max(workspace, 1), dtype=torch.uint8, device=device)
        _workspaces[key] = ws
    algo_buf = torch.frombuffer(bytearray(algo), dtype=torch.uint8).clone()

    def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device)
        with torch.cuda.device(x.device):
            handle = torch.cuda.current_blas_handle()
        module.mm_bf16_cublaslt_run_with_algo(x, weight, None, out, ws, handle, algo_buf, 0)
        return out

    return run


def _tgv_runner(tactic: int):
    from sglang.kernels.ops.gemm import cutedsl_bf16_gemm

    cutedsl_bf16_gemm._resolve_tactic(tactic)  # range check

    def run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device)
        cutedsl_bf16_gemm._run_tgv(x, weight.t(), None, out, True, tactic)
        return out

    return run


def _build(key, spec, weight: torch.Tensor, stock: Callable) -> Optional[Callable]:
    """The pinned runner for ``key`` if it launches and matches ``stock`` bit for bit, else None (for good)."""
    m, n, k = key[1:]
    try:
        if spec[0] == "cublaslt":
            run = _cublaslt_runner(weight.device, n, k, spec[1], spec[2])
        else:
            run = _tgv_runner(spec[1])
        gen = torch.Generator(device=weight.device).manual_seed(_CHECK_SEED + m)
        x = torch.randn((m, k), device=weight.device, generator=gen).to(torch.bfloat16)
        got, ref = run(x, weight), stock(x, weight)
        same = torch.equal(got.view(torch.int16), ref.view(torch.int16))
    except Exception as e:  # noqa: BLE001 - no module, bad algorithm or tactic: stock for good
        logger.warning("BF16 GEMM pin M=%d N=%d K=%d %s refused, stock kept: %r", m, n, k, spec[0], e)
        return None
    if not same:
        logger.warning("BF16 GEMM pin M=%d N=%d K=%d %s differs from stock, stock kept", m, n, k, spec[0])
        return None
    logger.info("BF16 GEMM pin engaged: M=%d N=%d K=%d %s%s", m, n, k, spec[0],
                f" tactic {spec[1]}" if spec[0] == "tgv" else "")
    return run


def gemm(x: torch.Tensor, weight: torch.Tensor, stock: Callable) -> Optional[torch.Tensor]:
    """``x @ weight.T`` on the pinned kernel for this (M, N, K), or None to run the stock dispatch.

    ``stock(x, weight)`` is the dispatch without pins; it is called only once per key, by the check."""
    state = _load()
    if not state or x.ndim != 2 or weight.ndim != 2:
        return None
    n, k = weight.shape
    if (n, k) not in state["shapes"]:
        return None
    m = x.shape[0]
    spec = state["entries"].get((m, n, k))
    if (
        spec is None
        or x.shape[1] != k
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or torch.compiler.is_compiling()
        or is_batch_invariant_mode_enabled()
    ):
        return None
    key = (x.device.index, m, n, k)
    if key in _runners:
        run = _runners[key]
    elif torch.cuda.is_current_stream_capturing():
        logger.warning("BF16 GEMM pin M=%d N=%d K=%d first seen under graph capture: stock in this graph", m, n, k)
        return None
    else:
        run = _runners[key] = _build(key, spec, weight, stock)
    return None if run is None else run(x, weight)
