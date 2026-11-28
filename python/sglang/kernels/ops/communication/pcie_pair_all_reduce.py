"""Launcher for the PCIe pair all-reduce (``csrc/distributed/pcie_pair_all_reduce.cuh``).

The communicator that owns the IPC buffers and picks the launch shape is
``sglang.srt.distributed.device_communicators.pcie_pair_ar``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "pcie_pair_all_reduce",
        cuda_files=["distributed/pcie_pair_all_reduce.cuh"],
        cuda_wrappers=[("run", "PciePairAllReduceKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def pcie_pair_all_reduce(
    inp: torch.Tensor,
    out: torch.Tensor,
    counters: torch.Tensor,
    *,
    self_buf: int,
    partner_buf: int,
    cross_buf: int,
    pos: int,
    max_blocks: int,
    half_cap: int,
    data_off: int,
    blocks: int,
    threads: int,
    one_shot: bool,
    wide: bool,
) -> None:
    """Sum ``inp`` over the 4 ranks into ``out`` (both contiguous bf16, same shape)."""
    _jit_module().run(
        inp.view(-1),
        out.view(-1),
        counters,
        self_buf,
        partner_buf,
        cross_buf,
        pos,
        max_blocks,
        half_cap,
        data_off,
        blocks,
        threads,
        one_shot,
        wide,
    )
