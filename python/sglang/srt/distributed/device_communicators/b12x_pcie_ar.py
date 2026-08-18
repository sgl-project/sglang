# SPDX-License-Identifier: Apache-2.0
"""b12x PCIe all-reduce communicator for SM120 (Blackwell workstation/server parts).

`b12x` (https://pypi.org/project/b12x/, Apache-2.0) is an SM120/SM121-only CuTe DSL
kernel library. This wrapper exposes its two IPC-backed all-reduce runtimes:

* ``OneshotAllReducePool`` -- one-shot all-reduce for small messages (decode-sized
  hidden states), plus a fused ``all_reduce + residual_add + RMSNorm`` entry point.
* ``DmaAllReduce`` -- copy-engine ring all-reduce with optional FP8 wire format
  for large messages (prefill chunks).

b12x moved both out of ``b12x.distributed`` into ``b12x.comm.pcie`` and dropped their
``PCIe`` prefix, the package path now carrying it. ``_import_pcie_all_reduce`` accepts
either layout; nothing else here changes, because the factory, the constructor keywords
and every method this wrapper calls kept their names across that move.

Both target hosts without NVLink, where NCCL all-reduce dominates the per-layer cost.
The kernels are opt-in via ``SGLANG_B12X_PCIE_AR`` / ``SGLANG_B12X_PCIE_DMA`` and every
dispatch site falls back to the existing NCCL path when a tensor is out of range.

Sizing the one-shot path
------------------------
One-shot is latency-optimal but bandwidth-poor: every rank writes its shard to all
``N - 1`` peers, so the number of flows crossing a NUMA/UPI hop grows as ``(N/2)^2``
-- 16 flows at ``N=8`` against 4 at ``N=4``. Below saturation one-shot wins by a wide
margin; once those flows saturate the hop it collapses and NCCL's ring wins. The
crossover is therefore a property of the deployment, not of the kernel. Measured on
one 8x SM120 host (GPUs 0-3 and 4-7 on separate NUMA nodes, ``SYS`` between them):

===================  ==================  =====================================
Group placement      Cross-hop flows     One-shot stays ahead of NCCL up to
===================  ==================  =====================================
4 ranks, one node    0                   ~512 KiB
4 ranks, 2+2 split   4                   >8 MiB (never lost in range)
8 ranks, 4+4 split   16                  ~16 KiB
===================  ==================  =====================================

The pool's own limit (8 MiB) sits above the crossover on every placement measured, so
``SGLANG_B12X_ONESHOT_MAX_BYTES`` exists to bound it. It defaults to ``UNCAPPED``
(unchanged behavior); ``AUTOTUNE`` measures the crossover at init instead of asking
operators to derive it from the table above. A single non-zero default would be wrong
somewhere -- capping at 64 KiB recovers a 2.9x decode regression on the 8-rank split
and costs 46% decode throughput on the 2+2 split.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.environ import B12xOneshotCap, envs

logger = logging.getLogger(__name__)

# b12x IPC channels are built for these world sizes only.
_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# The fused all-reduce + RMSNorm kernel is specialized for small token counts.
# Larger batches fall back to the unfused path.
_FUSED_MAX_ROWS = 36

# Autotune ladder. Geometric so a handful of points bracket the crossover on any
# placement; the walk stops at the first loss, so the common case is a few hundred
# microseconds of init. The top rung matches the pool limit so a placement where
# one-shot never loses can still use the whole range.
_PROBE_BYTES = (4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 8 << 20)
_PROBE_WARMUP = 3
_PROBE_ITERS = 10
# One-shot must beat NCCL by this margin to keep a size; without it the cap lands in
# the noise band where the two are interchangeable.
_PROBE_MARGIN = 1.05


def _import_pcie_all_reduce():
    """Return ``(OneshotAllReducePool, DmaAllReduce)`` from whichever layout is installed.

    b12x >= 1.0 serves them from ``b12x.comm.pcie`` without the ``PCIe`` prefix;
    older releases from ``b12x.distributed`` with it. Raises ``ImportError`` when
    neither is present, which the callers turn into a fall back to NCCL.
    """
    try:
        from b12x.comm.pcie import DmaAllReduce, OneshotAllReducePool
    except ImportError:
        from b12x.distributed import PCIeDmaAllReduce as DmaAllReduce
        from b12x.distributed import PCIeOneshotAllReducePool as OneshotAllReducePool

    return OneshotAllReducePool, DmaAllReduce


def _time_all_reduce(*, fn, buf: torch.Tensor, group: ProcessGroup) -> float:
    """Microseconds per all-reduce of ``buf``, averaged over the probe iterations."""
    for _ in range(_PROBE_WARMUP):
        fn(buf)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_PROBE_ITERS):
        fn(buf)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / _PROBE_ITERS * 1000.0


class B12xPCIeCommunicator:
    """Thin adapter between ``GroupCoordinator`` and the b12x all-reduce runtimes."""

    def __init__(self, group: ProcessGroup, device: torch.device | int):
        self.disabled = True
        self.max_size = 0
        self.oneshot_max_size = 0
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
            oneshot_pool_cls, _ = _import_pcie_all_reduce()
        except ImportError:
            logger.warning(
                "SGLANG_B12X_PCIE_AR is set but the `b12x` package is not installed; "
                "falling back to NCCL all-reduce."
            )
            return

        try:
            # NOTE: the IPC handshake requires a CUDA (NCCL) process group; passing the
            # gloo CPU group raises inside b12x.
            self._oneshot = oneshot_pool_cls.from_exchange_group(
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

        self.oneshot_max_size = self._resolve_oneshot_cap(group=group, device=device)
        self.disabled = False
        logger.info(
            "b12x PCIe one-shot all-reduce enabled "
            "(world=%d, max_size=%d bytes, oneshot_cap=%d bytes)",
            world_size,
            self.max_size,
            self.oneshot_max_size,
        )

        if envs.SGLANG_B12X_PCIE_DMA.get():
            self._init_dma(group, device)

    def _resolve_oneshot_cap(
        self, *, group: ProcessGroup, device: torch.device
    ) -> int:
        """Largest message the one-shot path may take, in bytes.

        See the module docstring for why this is deployment-specific. A cap of 0 leaves
        one-shot unused, which is a legitimate autotune outcome on a saturated hop.
        """
        cap = envs.SGLANG_B12X_ONESHOT_MAX_BYTES.get()
        if cap == B12xOneshotCap.AUTOTUNE:
            return self._autotune_oneshot_cap(group=group, device=device)
        if cap == B12xOneshotCap.UNCAPPED:
            return self.max_size
        return min(self.max_size, cap)

    def _autotune_oneshot_cap(
        self, *, group: ProcessGroup, device: torch.device
    ) -> int:
        """Walk the probe ladder and cap at the last size where one-shot still wins.

        Every rank runs the identical ladder, so the collectives stay in lockstep. The
        walk stops at the first loss: one-shot degrades monotonically past saturation,
        so continuing would only add init latency.
        """
        try:
            cap = 0
            for nbytes in _PROBE_BYTES:
                if nbytes > self.max_size:
                    break
                buf = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=device)
                nccl_us = _time_all_reduce(
                    fn=lambda t: dist.all_reduce(t, group=group), buf=buf, group=group
                )
                b12x_us = _time_all_reduce(
                    fn=self._oneshot.all_reduce, buf=buf, group=group
                )
                if b12x_us * _PROBE_MARGIN > nccl_us:
                    break
                cap = nbytes
            logger.info("b12x one-shot autotune selected cap=%d bytes", cap)
            return cap
        except Exception as e:
            logger.warning(
                "b12x one-shot autotune failed: %s; keeping the pool limit", e
            )
            return self.max_size

    def _init_dma(self, group: ProcessGroup, device: torch.device) -> None:
        try:
            _, dma_cls = _import_pcie_all_reduce()

            max_bytes = envs.SGLANG_B12X_DMA_MAX_BYTES.get()
            dma = dma_cls(
                exchange_group=group,
                device=device,
                max_bytes=max_bytes,
                fp8=envs.SGLANG_B12X_DMA_FP8.get() or None,
            )
            # The DMA ring only covers what the one-shot *kernel* cannot, so this keys
            # off the pool limit and not off oneshot_max_size: a one-shot cap hands the
            # sizes it gives up back to NCCL, which beats the ring there. The ring is
            # fixed-cost dominated (~1ms regardless of size below a few MiB), so pulling
            # its floor down to the cap would be strictly worse than the NCCL fallback.
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
        return inp.numel() * inp.element_size() <= self.oneshot_max_size

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
