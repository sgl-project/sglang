"""MIDQ: FlashInfer MoE all-to-all built from a patched copy of its kernel source in which
the combine splits each token row over several CTAs. Every output element is computed by the
same instruction sequence as the stock kernel, so the combine output is bit-identical."""

from __future__ import annotations

import logging
import pathlib
import threading
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_KERNEL_SOURCE = (
    pathlib.Path(__file__).resolve().parent / "midq_csrc" / "moeAlltoAllKernels_midq.cu"
)
_installed_split: Optional[int] = None
_stock_module = None


def _split_combine_spec(split: int):
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec

    # Same sources and flags as FlashInfer 0.7.1 gen_moe_alltoall_module("legacy"),
    # with moeAlltoAllKernels.cu replaced by the patched copy.
    csrc = jit_env.FLASHINFER_CSRC_DIR
    common = csrc / "nv_internal" / "cpp" / "common"
    return gen_jit_spec(
        f"mnnvl_moe_alltoall_midq_s{split}",
        [
            csrc / "trtllm_moe_alltoall.cu",
            _KERNEL_SOURCE,
            common / "envUtils.cpp",
            common / "tllmException.cpp",
            common / "logger.cpp",
            common / "stringUtils.cpp",
        ],
        extra_include_paths=[
            str(csrc / "nv_internal"),
            str(csrc / "nv_internal" / "include"),
        ],
        extra_cuda_cflags=["-DENABLE_BF16", f"-DMIDQ_COMBINE_SPLIT={split}"],
    )


def install_split_combine(split: int) -> None:
    """Make FlashInfer's default (trtllm) MoE a2a module the split-combine build.

    Must run before the process loads the stock module; dispatch, combine and the
    workspace layout then all come from one module.
    """
    global _installed_split
    if _installed_split == split:
        return
    import flashinfer.comm.trtllm_moe_alltoall as fi_a2a

    if fi_a2a._get_moe_alltoall_module_for_target.cache_info().currsize:
        raise RuntimeError(
            "Split combine must be installed before FlashInfer loads its "
            "MoE all-to-all module"
        )
    stock_gen = fi_a2a.gen_moe_alltoall_module

    def gen(target="legacy"):
        return _split_combine_spec(split) if target == "legacy" else stock_gen(target)

    fi_a2a.gen_moe_alltoall_module = gen
    _installed_split = split
    logger.info(
        "MIDQ FlashInfer a2a combine: split-token build mnnvl_moe_alltoall_midq_s%d installed",
        split,
    )


def stock_combine_module():
    """The unpatched FlashInfer 0.7.1 module, loaded raw (no custom-op registration)."""
    global _stock_module
    if _stock_module is None:
        from flashinfer.jit.comm import gen_moe_alltoall_module

        _stock_module = gen_moe_alltoall_module("legacy").build_and_load()
    return _stock_module


class CombineProof:
    """Check that the installed combine equals the stock kernel bit for bit, eager or in CUDA graphs.

    One instance per process and per (target, draft); every dispatcher of that kind shares it.
    Counters live on the device and are mirrored into pinned host memory by a copy that is part
    of the check, so the reporter thread logs them every PERIOD_S seconds without any CUDA call
    (a CUDA call from another thread can disturb a stream capture).
    """

    PERIOD_S = 30.0

    def __init__(self, speculative: bool):
        self.speculative = speculative
        # [calls, compared elements, differing elements]
        self.counters = torch.zeros(
            3, dtype=torch.int64, device=torch.cuda.current_device()
        )
        # Model init runs under a CUDA default-device context; pinned memory must say cpu.
        self.host_counters = torch.zeros(
            3, dtype=torch.int64, device="cpu", pin_memory=True
        )
        threading.Thread(target=self._report, daemon=True).start()

    def check(self, ref: torch.Tensor, out: torch.Tensor) -> None:
        differing = torch.count_nonzero(ref.view(torch.int16) != out.view(torch.int16))
        self.counters[0].add_(1)
        self.counters[1].add_(ref.numel())
        self.counters[2].add_(differing)
        self.host_counters.copy_(self.counters, non_blocking=True)

    def _report(self) -> None:
        while True:
            time.sleep(self.PERIOD_S)
            calls, elements, differing = self.host_counters.tolist()
            logger.info(
                "MIDQ combine proof: speculative=%s calls=%d elements=%d mismatched=%d",
                self.speculative,
                calls,
                elements,
                differing,
            )


_proofs: dict[bool, CombineProof] = {}


def get_combine_proof(speculative: bool) -> CombineProof:
    if speculative not in _proofs:
        _proofs[speculative] = CombineProof(speculative)
    return _proofs[speculative]
