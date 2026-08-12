# SPDX-License-Identifier: Apache-2.0
"""FlashInfer PCIe-IPC all-reduce for switch-free intra-node machines.

FlashInfer's ``pcie_ipc_comm`` targets hosts where every peer transfer crosses the
CPU root complex -- no NVLink, no multicast. The existing custom all-reduce
backends assume one of those fabrics, so on such a host every per-layer reduction
falls back to NCCL, which leaves a large margin on the table at decode sizes.

Why the kernel wins here
------------------------
Naive all-to-all peer writes collapse on this fabric, and the collapse worsens
with the number of concurrently writing blocks. FlashInfer's kernels instead
*stage* their pushes so each rank has exactly one outbound and one inbound stream
at any instant, and the 8-rank path keeps a 4+4 island decomposition so the
scarce cross-socket link carries the minimum. Measured on one 8x SM120 host
(GPUs 0-3 and 4-7 on separate NUMA nodes), against NCCL, hidden 6144 bf16:

=========  ===========  ==========  =========
Message    FlashInfer   NCCL        Speed-up
=========  ===========  ==========  =========
12 KiB     5.3 us       205.1 us    38.8x
96 KiB     15.3 us      584.9 us    38.2x
384 KiB    34.6 us      1169.3 us   33.8x
100 MiB    8.5 ms       50.4 ms     5.9x
=========  ===========  ==========  =========

Shape coverage is decided by FlashInfer's own tuning table, reached through
:meth:`PcieIpcAllReduceWorkspace.supports`. A shape the kernels do not beat is
reported unsupported and falls back to NCCL, so this wrapper needs no size knob
of its own -- it asks and obeys.

Workspace sizing
----------------
The workspace cannot grow after construction and costs roughly
``2 * world_size * max_numel * itemsize`` bytes per rank, so it has to be sized
for the largest reduction the model will issue: one prefill chunk, i.e.
``chunked_prefill_size * hidden``. Sizing it for decode instead is a false
economy -- it saves ~3 GiB of KV but leaves the prefill reduction on NCCL, and
measured end to end that costs more than the memory is worth: at 8 ranks, TP8,
8k context, TTFT was 2.2 s with a prefill-sized workspace against 12.0 s with a
decode-sized one, where NCCL alone was 8.5 s. Splitting reductions across two
paths is worse than committing to either.

``chunked_prefill_size`` comes from the server args, but the hidden size is not
known when the group is built, so the workspace is created on the first eligible
tensor, whose trailing dimension is exactly that hidden size. Ranks run the same
sequence of reductions, so they all reach that first call with the same shape and
build the same workspace. ``SGLANG_PCIE_IPC_MAX_NUMEL`` overrides the derivation
for deployments that would rather cap the memory and leave large reductions on
NCCL.
"""

import logging
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# The kernels build IPC channels for these world sizes only.
_SUPPORTED_WORLD_SIZES = (2, 4, 8)

# Fallback when the server args do not carry a chunk size (unit tests, embedded
# use). Large enough for any decode shape we serve; prefill stays on NCCL.
_FALLBACK_MAX_NUMEL = 64 * 8192


def _chunked_prefill_size() -> Optional[int]:
    """Tokens in one prefill forward, or None when the server args are absent."""
    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args().chunked_prefill_size
    except Exception:
        return None


class PcieIpcCommunicator:
    """Adapter between ``GroupCoordinator`` and FlashInfer's PCIe-IPC all-reduce."""

    def __init__(self, group: ProcessGroup, device: torch.device | int):
        self.disabled = True
        self.max_numel = 0
        self._workspace: Optional[Any] = None
        self._bound_stream: Optional[torch.cuda.Stream] = None

        world_size = dist.get_world_size(group=group)
        if world_size not in _SUPPORTED_WORLD_SIZES:
            logger.debug(
                "FlashInfer PCIe-IPC all-reduce skipped: unsupported world size %d",
                world_size,
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        if device.type != "cuda":
            return

        try:
            from flashinfer.comm import PcieIpcAllReduceWorkspace
        except ImportError:
            logger.warning(
                "SGLANG_ENABLE_PCIE_IPC_ALLREDUCE is set but this FlashInfer build has "
                "no pcie_ipc_comm; falling back to NCCL all-reduce."
            )
            return

        # The workspace is built on the first eligible tensor: sizing it needs the
        # hidden size, which the group does not know yet.
        self._group = group
        self._device = device
        self._world_size = world_size
        self._workspace_cls = PcieIpcAllReduceWorkspace
        self._build_failed = False
        self.disabled = False
        logger.info(
            "FlashInfer PCIe-IPC all-reduce enabled (world=%d, workspace sized on "
            "first reduction)",
            world_size,
        )

    def _ensure_workspace(self, inp: torch.Tensor) -> bool:
        """Build the workspace for one prefill chunk, using ``inp`` for the hidden size.

        Every rank issues the same reductions in the same order, so they all arrive
        here with the same shape and agree on ``max_numel`` without extra exchange.
        """
        if self._workspace is not None:
            return True
        if self._build_failed:
            return False

        override = envs.SGLANG_PCIE_IPC_MAX_NUMEL.get()
        if override:
            max_numel = override
        else:
            hidden = inp.shape[-1]
            chunk = _chunked_prefill_size()
            max_numel = chunk * hidden if chunk else _FALLBACK_MAX_NUMEL

        try:
            self._workspace = self._workspace_cls(
                group=self._group, max_numel=max_numel, dtype=torch.bfloat16
            )
        except Exception as e:
            logger.warning(
                "FlashInfer PCIe-IPC workspace (max_numel=%d) failed: %s; "
                "reductions stay on NCCL",
                max_numel,
                e,
            )
            self._build_failed = True
            self.disabled = True
            return False

        self.max_numel = max_numel
        logger.info(
            "FlashInfer PCIe-IPC workspace built (world=%d, max_numel=%d, "
            "hidden=%d)",
            self._world_size,
            max_numel,
            inp.shape[-1],
        )
        return True

    def should_pcie_ipc_ar(self, inp: torch.Tensor) -> bool:
        """Whether FlashInfer has a tuned configuration for this exact shape.

        ``supports`` consults the tuning table, so a shape the kernels lose on is
        rejected here and the caller keeps its NCCL path.
        """
        if self.disabled or not inp.is_contiguous() or inp.dim() < 2:
            return False
        if not self._ensure_workspace(inp):
            return False
        if inp.numel() > self.max_numel:
            return False
        return self._workspace.supports(inp)

    def pcie_ipc_all_reduce(self, inp: torch.Tensor) -> Optional[torch.Tensor]:
        """All-reduce ``inp``, rebinding the workspace when the stream changes.

        One workspace serves one stream: its epoch and arrival counters assume the
        calls sharing it are totally ordered. SGLang switches streams at phase
        boundaries -- graph capture, then replay -- so the binding is moved with
        the caller rather than kept on whichever stream got there first. This is
        safe because those phases do not overlap; it would not be safe if two
        streams issued reductions concurrently on the same group.
        """
        stream = torch.cuda.current_stream()
        if self._bound_stream is None or stream != self._bound_stream:
            self._workspace.rebind_stream()
            self._bound_stream = stream
        return self._workspace.all_reduce(inp)

    def destroy(self) -> None:
        if self._workspace is not None:
            self._workspace.destroy()
            self._workspace = None
            self.disabled = True
