"""Private sm107a LM-head plans: FP32 accumulate, one final BF16 rounding."""
from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import logging
import os
from pathlib import Path

import torch

_SCOPE = os.environ.get("SGLANG_Q38C_LM_HEAD_SCOPE", "draft")
_LOGGED = set()


class _Plan:
    def __init__(self, root: str, device: torch.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("LM-head plan must be warmed before graph capture")
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"LM-head artifact SHA mismatch: {name}")
        self.cases = json.loads((root / "plan.json").read_text())["cases"]
        self.device = device
        self.contexts = {}
        self.workspaces = {}
        self.lib = ctypes.CDLL(str(root / "lt_probe.so"))
        self.lib.lm_lt_version.restype = ctypes.c_size_t
        if self.lib.lm_lt_version() != manifest["cublasLt_version"]:
            raise RuntimeError("Frozen LM-head cuBLASLt version mismatch")
        self.lib.lm_error.restype = ctypes.c_char_p
        self.lib.lm_create_fixed.argtypes = [ctypes.c_int] * 3 + [ctypes.c_size_t]
        self.lib.lm_create_fixed.restype = ctypes.c_void_p
        self.lib.lm_import.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.lm_algo_size.restype = ctypes.c_size_t
        self.lib.lm_workspace.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.lm_workspace.restype = ctypes.c_size_t
        self.lib.lm_run_sized.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 4 + [ctypes.c_size_t, ctypes.c_void_p]
        from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk as warp

        if hashlib.sha256(Path(warp.__file__).read_bytes()).hexdigest() != manifest["FI_warp_source_SHA256"]:
            raise RuntimeError("LM-head FlashInfer source version mismatch")
        self.warp = warp

    def run(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        case = self.cases[str(m)]
        out = torch.empty((m, 62080), dtype=torch.bfloat16, device=x.device)
        if case["backend"] == "FI_warp":
            tactic = self.warp.WarpSplitKTactic(*case["tactic"])
            # Direct call bypasses FlashInfer's autotuner and its cache loader.
            self.warp.run_warp_splitk_dense(x, weight.T, out, True, tactic)
        else:
            stream = torch.cuda.current_stream(x.device).cuda_stream
            if m not in self.contexts:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(f"LM-head M{m} must be warmed before capture")
                ctx = self.lib.lm_create_fixed(m, 62080, 2560, 64 * 1024**2)
                if not ctx:
                    raise RuntimeError(self.lib.lm_error().decode())
                raw = bytes.fromhex(case["algorithm_hex"])
                if len(raw) != self.lib.lm_algo_size():
                    raise RuntimeError("LM-head cuBLASLt algorithm ABI mismatch")
                blob = ctypes.create_string_buffer(raw)
                index = self.lib.lm_import(ctx, blob)
                if index < 0:
                    raise RuntimeError("Frozen non-split LM-head algorithm rejected")
                required = self.lib.lm_workspace(ctx, index)
                if required > 64 * 1024**2:
                    raise RuntimeError("LM-head workspace query failed or exceeds qualified limit")
                self.contexts[m] = ctx, index, required
            ctx, index, required = self.contexts[m]
            key = m, stream
            if required and key not in self.workspaces:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("LM-head workspace must be warmed before capture")
                # Workspaces are private to a stream; ordered calls reuse it.
                self.workspaces[key] = torch.empty(required, dtype=torch.uint8, device=x.device)
            workspace = self.workspaces[key].data_ptr() if required else 0
            rc = self.lib.lm_run_sized(ctx, index, weight.data_ptr(), x.data_ptr(), out.data_ptr(), workspace, required, stream)
            if rc:
                raise RuntimeError(self.lib.lm_error().decode())
        return out


@functools.lru_cache(None)
def _plan(root: str, device: torch.device) -> _Plan:
    return _Plan(root, device)


def try_run_lm_head(x, weight, *, artifacts: str, is_draft: bool):
    if _SCOPE not in ("draft", "all"):
        raise ValueError(f"Unsupported LM-head scope: {_SCOPE}")
    if _SCOPE == "draft" and not is_draft:
        return None
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.device != x.device
        or x.ndim != 2
        or weight.shape != (62080, 2560)
        or x.shape[1] != 2560
        or x.shape[0] not in (1, 2, 4, 8)
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or torch.cuda.get_device_capability(x.device) != (10, 7)
    ):
        return None
    plan = _plan(artifacts, x.device)
    key = is_draft, x.shape[0]
    if key not in _LOGGED:
        logging.getLogger(__name__).info(
            "RUBIN_LM_HEAD_ENGAGED role=%s M=%s plan=%s arch=sm107a",
            "draft" if is_draft else "verify", x.shape[0], plan.cases[str(x.shape[0])],
        )
        _LOGGED.add(key)
    return plan.run(x, weight)
