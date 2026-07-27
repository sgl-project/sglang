# SPDX-License-Identifier: Apache-2.0
"""b12x PCIe all-reduce communicator for SM120 (Blackwell workstation/server parts).

`b12x` (https://pypi.org/project/b12x/, Apache-2.0) is an SM120/SM121-only CuTe DSL
kernel library. This wrapper exposes its two IPC-backed all-reduce runtimes:

* ``PCIeOneshotAllReducePool`` -- one-shot all-reduce for small messages (decode-sized
  hidden states), plus a fused ``all_reduce + residual_add + RMSNorm`` entry point.
* ``PCIeDmaAllReduce`` -- copy-engine ring all-reduce with optional FP8 wire format
  for large messages (prefill chunks).

Both target hosts without NVLink, where NCCL all-reduce dominates the per-layer cost.
The kernels are opt-in via ``SGLANG_B12X_PCIE_AR`` / ``SGLANG_B12X_PCIE_DMA`` and every
dispatch site falls back to the existing NCCL path when a tensor is out of range.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# b12x IPC channels are built for these world sizes only.
_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# The fused all-reduce + RMSNorm kernel is specialized for small token counts.
# Larger batches fall back to the unfused path.
_FUSED_MAX_ROWS = 36


class B12xPCIeCommunicator:
    """Thin adapter between ``GroupCoordinator`` and the b12x all-reduce runtimes."""

    def __init__(self, group: ProcessGroup, device: torch.device | int):
        self.disabled = True
        self.max_size = 0
        self._oneshot = None
        self._dma = None

        world_size = dist.get_world_size(group=group)
        if world_size not in _SUPPORTED_WORLD_SIZES:
            logger.debug(
                "b12x PCIe all-reduce skipped: unsupported world size %d", world_size
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        if device.type != "cuda":
            return

        try:
            from b12x.distributed import PCIeOneshotAllReducePool
        except ImportError:
            logger.warning(
                "SGLANG_B12X_PCIE_AR is set but the `b12x` package is not installed; "
                "falling back to NCCL all-reduce."
            )
            return

        try:
            # NOTE: the IPC handshake requires a CUDA (NCCL) process group; passing the
            # gloo CPU group raises inside b12x.
            self._oneshot = PCIeOneshotAllReducePool.from_exchange_group(
                exchange_group=group, device=device
            )
            # Channels are stream-affine: bind one for the current stream up front so
            # CUDA graph capture does not hit an unbound channel.
            self._oneshot.for_stream()
            self.max_size = int(self._oneshot.max_size)
        except Exception as e:
            logger.warning("b12x PCIe one-shot all-reduce init failed: %s", e)
            self._oneshot = None
            return

        self.disabled = False
        logger.info(
            "b12x PCIe one-shot all-reduce enabled (world=%d, max_size=%d bytes)",
            world_size,
            self.max_size,
        )

        if envs.SGLANG_B12X_PCIE_DMA.get():
            self._init_dma(group, device)

    def _init_dma(self, group: ProcessGroup, device: torch.device) -> None:
        try:
            from b12x.distributed import PCIeDmaAllReduce

            max_bytes = envs.SGLANG_B12X_DMA_MAX_BYTES.get()
            dma = PCIeDmaAllReduce(
                exchange_group=group,
                device=device,
                max_bytes=max_bytes,
                fp8=envs.SGLANG_B12X_DMA_FP8.get() or None,
            )
            # The DMA ring only covers what the one-shot path cannot.
            dma.min_bytes = self.max_size + 1
            self._dma = dma
            logger.info(
                "b12x PCIe DMA all-reduce enabled (%s, %d..%d bytes)",
                dma.wire_mode,
                dma.min_bytes,
                max_bytes,
            )
        except Exception as e:
            logger.warning(
                "b12x PCIe DMA all-reduce init failed: %s; large all-reduces stay on NCCL",
                e,
            )
            self._dma = None

    # ---- one-shot (small messages) ----------------------------------------

    def should_b12x_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype not in _SUPPORTED_DTYPES or not inp.is_contiguous():
            return False
        return inp.numel() * inp.element_size() <= self.max_size

    def b12x_all_reduce(self, inp: torch.Tensor) -> Optional[torch.Tensor]:
        return self._oneshot.all_reduce(inp)

    # ---- fused all-reduce + residual add + RMSNorm ------------------------

    def should_b12x_fused_rmsnorm(self, inp: torch.Tensor) -> bool:
        return (
            self.should_b12x_ar(inp)
            and inp.dim() >= 2
            and inp.shape[0] <= _FUSED_MAX_ROWS
        )

    def fused_allreduce_rmsnorm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._oneshot.all_reduce_fused_add_rms_norm(inp, residual, weight, eps)

    # ---- DMA ring (large messages) ----------------------------------------

    def should_b12x_dma(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._dma is None:
            return False
        return self._dma.should_allreduce(inp)

    def b12x_dma_all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return self._dma.all_reduce(inp)
