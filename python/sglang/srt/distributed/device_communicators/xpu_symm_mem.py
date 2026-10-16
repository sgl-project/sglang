"""Symmetric-memory all-reduce for Intel XPU."""

import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.all_reduce_utils import (
    XPU_SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from sglang.srt.distributed.device_communicators.symm_mem_workspace import (
    SymmMemWorkspace,
    all_ranks_agree,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu

logger = logging.getLogger(__name__)


class XpuSymmMemCommunicator:
    """One-shot all-reduce over torch symmetric memory on Intel XPU.

    A separate class from ``TorchSymmMemCommunicator``, which is built on
    ``symm_mem::{multimem,two_shot}_all_reduce_``: torch-xpu-ops registers
    neither, and Xe reports ``multicast_ptr == 0``. Only the allocator half of
    the backend exists for XPU, hence the Triton reduce kernel.
    """

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        self.disabled = True
        self.workspace: Optional[SymmMemWorkspace] = None
        self.max_bytes = 0
        self._staging = {}
        self._counter: Optional[torch.Tensor] = None
        self._snapshot: Optional[torch.Tensor] = None
        if isinstance(device, int):
            device = torch.device(f"xpu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(group)

        if not is_xpu():
            return

        # Lazy: the reduce kernel pulls in Triton.
        from sglang.srt.hardware_backend.xpu.kernels.comm.one_shot_all_reduce import (
            SUPPORTED_DTYPES,
        )

        max_bytes = envs.SGLANG_XPU_SYMM_MEM_MAX_BYTES.get()
        if max_bytes is None:
            max_bytes = XPU_SYMM_MEM_ALL_REDUCE_MAX_SIZES.get(self.world_size, 0)
        # The buffer is reinterpreted as every supported dtype, so keep it
        # aligned to the widest element (fp32).
        self.max_bytes = max_bytes - max_bytes % 4
        if self.max_bytes <= 0:
            logger.info(
                "XpuSymmMemCommunicator: no payload ceiling for world size %d "
                "(SGLANG_XPU_SYMM_MEM_MAX_BYTES=%s); communicator is not available.",
                self.world_size,
                envs.SGLANG_XPU_SYMM_MEM_MAX_BYTES.get(),
            )
            return

        # Two staging halves of max_bytes each; consecutive calls alternate.
        workspace = SymmMemWorkspace.create(
            group=group,
            device=device,
            nbytes=2 * self.max_bytes,
            owner="XpuSymmMemCommunicator",
        )
        self._counter = torch.zeros(1, dtype=torch.int32, device=device)
        self._snapshot = torch.zeros(1, dtype=torch.int32, device=device)
        # Agreed on every rank, including one whose rendezvous failed alone:
        # ranks that disagree would diverge on every later all-reduce.
        ready = workspace is not None and self._compile_kernel(workspace)
        if not all_ranks_agree(ready, group):
            return

        self.workspace = workspace
        self._staging = {
            dtype: workspace.buffer.view(dtype) for dtype in SUPPORTED_DTYPES
        }
        self.disabled = False
        logger.info(
            "XpuSymmMemCommunicator: enabled on %s (world_size=%d, max_bytes=%d).",
            device,
            self.world_size,
            self.max_bytes,
        )

    def _compile_kernel(self, workspace: SymmMemWorkspace) -> bool:
        """Build both kernels for every supported dtype up front, so a build
        failure becomes a fallback here instead of an error on a forward pass.

        No barrier: the ranks have not yet agreed the workspace is usable, and
        a device barrier waiting on a rank still in rendezvous spins forever.
        Every rank runs the same calls, so the call counters stay in step.
        """
        from sglang.srt.hardware_backend.xpu.kernels.comm.one_shot_all_reduce import (
            SUPPORTED_DTYPES,
            one_shot_all_reduce,
            stage,
        )

        try:
            for dtype in SUPPORTED_DTYPES:
                probe = torch.zeros(64, dtype=dtype, device=self.device)
                staging = workspace.buffer.view(dtype)
                stage(
                    inp=probe,
                    staging=staging,
                    counter=self._counter,
                    snapshot=self._snapshot,
                )
                # Reads whatever peers hold; only the build matters here.
                one_shot_all_reduce(
                    peer_ptrs=workspace.peer_ptrs,
                    out=probe,
                    world_size=self.world_size,
                    half_numel=staging.numel() // 2,
                    counter=self._counter,
                    snapshot=self._snapshot,
                )
            torch.xpu.synchronize()
        except Exception as e:
            logger.warning(
                "XpuSymmMemCommunicator: one-shot kernel failed to build (%s); "
                "falling back to the default all-reduce.",
                e,
            )
            return False
        return True

    def should_torch_symm_mem_allreduce(self, inp: torch.Tensor) -> bool:
        """Whether the one-shot path can take this tensor."""
        if self.disabled:
            return False
        if inp.device != self.device:
            return False
        if inp.dtype not in self._staging:
            return False
        if not inp.is_contiguous():
            return False
        inp_bytes = inp.numel() * inp.element_size()
        if inp_bytes % 4 != 0:
            return False
        return inp_bytes <= self.max_bytes

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Sum-all-reduce ``inp`` by reading every rank's buffer once.

        Returns ``None`` for an ineligible input, which sends the caller to the
        default collective. ``out=inp`` reduces in place: peers read the staging
        copy, not ``out``.
        """
        if not self.should_torch_symm_mem_allreduce(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)

        self._stage_and_reduce(
            workspace=self.workspace,
            inp=inp.view(-1),
            out=out.view(-1),
            staging=self._staging[inp.dtype],
        )
        return out

    def _stage_and_reduce(
        self,
        *,
        workspace: SymmMemWorkspace,
        inp: torch.Tensor,
        out: torch.Tensor,
        staging: torch.Tensor,
    ) -> None:
        from sglang.srt.hardware_backend.xpu.kernels.comm.one_shot_all_reduce import (
            one_shot_all_reduce,
            stage,
        )

        stage(inp=inp, staging=staging, counter=self._counter, snapshot=self._snapshot)
        # Publishes every rank's input before anyone reads its peers. The next
        # call stages into the other half, so no second barrier is needed.
        workspace.barrier()
        one_shot_all_reduce(
            peer_ptrs=workspace.peer_ptrs,
            out=out,
            world_size=self.world_size,
            half_numel=staging.numel() // 2,
            counter=self._counter,
            snapshot=self._snapshot,
        )
