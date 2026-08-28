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
The workspace is sized for decode, ``cuda_graph_max_bs * hidden``, which leaves
every prefill chunk on NCCL. That split is deliberate. These kernels win by
latency at small messages; a prefill chunk is three orders of magnitude larger,
and there NCCL's ring is the better algorithm. Measured at 8 ranks, TP8, 8k
context, against NCCL as the baseline:

===================  ==========  ==========  ===============
Workspace            TTFT        TPOT        Output tok/s
===================  ==========  ==========  ===============
NCCL only            1910 ms     21.14 ms    35.02
prefill-sized        3176 ms     13.64 ms    38.43
decode-sized         1849 ms     13.62 ms    48.05
===================  ==========  ==========  ===============

Handing prefill to these kernels costs 66% on TTFT and buys nothing on TPOT.
The decode-sized workspace keeps the whole TPOT win, stays level with NCCL on
TTFT, and is the only one of the three that is not worse than NCCL anywhere we
measured. It is also ~250x smaller, which matters at long context: at 128k with
4 concurrent requests the prefill-sized workspace regressed TPOT by 45%, and
that regression disappears when it is sized for decode.

The bound comes from the server args, but the hidden size is not known when the
group is built, so the workspace is created on the first eligible tensor, whose
trailing dimension is exactly that hidden size. Ranks run the same sequence of
reductions, so they all reach that first call with the same shape and build the
same workspace. ``SGLANG_PCIE_IPC_MAX_NUMEL`` overrides the derivation for
deployments that want to hand larger reductions to the kernels anyway.
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

# Fallback decode width when the server args are absent (unit tests, embedded
# use). Covers the batch sizes we serve; anything larger stays on NCCL.
_FALLBACK_DECODE_WIDTH = 64


def _decode_width() -> Optional[int]:
    """Rows in the widest decode reduction, or None when the server args are absent.

    Decode runs inside a captured graph, so the largest captured batch bounds the
    reduction. Speculative decoding verifies several tokens per sequence in one
    forward, and the decode phase's ``max_bs`` is the batch the runner captures
    for, which already accounts for that.
    """
    try:
        from sglang.srt.server_args import get_global_server_args

        config = get_global_server_args().cuda_graph_config
        return config.decode.max_bs if config is not None else None
    except Exception:
        return None


def _tune_cache_path(world_size: int) -> Optional[str]:
    """Where FlashInfer should persist the measured tactics.

    Named explicitly rather than left to default, because the tuning run has to
    write the file for the *next* server to reuse it; a workspace built without
    a cache path measures every start.
    """
    try:
        from flashinfer.comm.pcie_ipc_tuning import default_cache_path

        return str(default_cache_path(world_size))
    except (AttributeError, ImportError):
        return None


def _tune_batches(max_rows: int) -> tuple[int, ...]:
    """Row counts to profile, bounded by what the workspace can actually take.

    The autotuner measures one tactic per shape, so profiling rows the workspace
    would reject is wasted startup time. Powers of two cover the captured decode
    batches; ``max_rows`` is appended because it is the widest reduction served
    and is not always a power of two.
    """
    rows = [r for r in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if r <= max_rows]
    if max_rows >= 1 and max_rows not in rows:
        rows.append(max_rows)
    return tuple(rows)


class PcieIpcCommunicator:
    """Adapter between ``GroupCoordinator`` and FlashInfer's PCIe-IPC all-reduce."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device | int,
        cpu_group: Optional[ProcessGroup] = None,
    ):
        self.disabled = True
        self.max_numel = 0
        self._workspace: Optional[Any] = None
        self._bound_stream: Optional[torch.cuda.Stream] = None
        self._cpu_group = cpu_group

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
        """Build the workspace for the widest decode, using ``inp`` for the hidden size.

        Every rank issues the same reductions in the same order, so they all arrive
        here with the same shape and agree on ``max_numel`` without extra exchange.
        """
        if self._workspace is not None:
            return True
        if self._build_failed:
            return False

        hidden = inp.shape[-1]
        override = envs.SGLANG_PCIE_IPC_MAX_NUMEL.get()
        if override:
            max_numel = override
        else:
            max_numel = (_decode_width() or _FALLBACK_DECODE_WIDTH) * hidden

        try:
            self._workspace = self._workspace_cls(
                group=self._group,
                max_numel=max_numel,
                dtype=torch.bfloat16,
                tune_batches=_tune_batches(max_numel // hidden),
                tune_cache=_tune_cache_path(self._world_size),
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
            hidden,
        )
        self._tune(hidden)
        return True

    def prepare(self, hidden: int) -> None:
        """Build and measure ahead of the first reduction.

        Call this from warmup, before any other autotuning starts. Left to the
        first reduction instead, the build lands inside SGLang's own FlashInfer
        autotune pass, and FlashInfer refuses to profile a collective from
        inside an autotune context it did not open -- so the tuning call would
        return having measured nothing.
        """
        if self.disabled or self._workspace is not None:
            return
        probe = torch.empty((1, hidden), dtype=torch.bfloat16, device=self._device)
        self._ensure_workspace(probe)

    def _tune(self, hidden: int) -> None:
        """Measure the launch tactics for this workspace's shapes.

        Without this the kernels run FlashInfer's seed policy, and ``supports``
        answers from that policy rather than from measurements on this host.
        Results persist to FlashInfer's cache, so only the first server pays.
        """
        from flashinfer.autotuner import AutoTuner

        if AutoTuner.get().is_tuning_mode:
            # FlashInfer checks the *installed* autotune process group, which
            # belongs to whoever opened the enclosing context, so it declines to
            # profile and the call would measure nothing. Say so rather than
            # report a tuning that did not happen.
            logger.warning(
                "FlashInfer PCIe-IPC all-reduce reached its first reduction inside "
                "another autotune context; skipping autotune and keeping the seed "
                "policy. Call PcieIpcCommunicator.prepare() from warmup to tune."
            )
            return
        if self._cpu_group is None:
            logger.warning(
                "FlashInfer PCIe-IPC all-reduce has no host group to autotune on; "
                "running FlashInfer's seed policy instead of measured tactics."
            )
            return
        if torch.cuda.is_current_stream_capturing():
            logger.warning(
                "FlashInfer PCIe-IPC all-reduce reached its first reduction inside "
                "a graph capture; skipping autotune and keeping the seed policy."
            )
            return
        try:
            tuned = self._workspace.tune(
                [hidden], dtype=torch.bfloat16, tune_group=self._cpu_group
            )
        except Exception as e:
            logger.warning(
                "FlashInfer PCIe-IPC autotune failed (%s); keeping the seed policy.",
                e,
            )
            return
        # Report what tune() covered, not that it returned: it declines shapes
        # silently, and a log line that only proves no exception was raised is
        # how an autotune that measured nothing reads as a success.
        if not tuned:
            logger.warning(
                "FlashInfer PCIe-IPC autotune covered no shapes for hidden=%d "
                "(requested rows %s); the kernels keep the seed policy.",
                hidden,
                _tune_batches(self.max_numel // hidden),
            )
            return
        logger.info(
            "FlashInfer PCIe-IPC all-reduce autotuned %d shape(s) (hidden=%d, rows=%s)",
            len(tuned),
            hidden,
            _tune_batches(self.max_numel // hidden),
        )

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
