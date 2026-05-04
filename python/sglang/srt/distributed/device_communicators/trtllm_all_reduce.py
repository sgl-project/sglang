# SPDX-License-Identifier: Apache-2.0

import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()


class TRTLLMAllReduce:
    """All-reduce backend that dispatches to flashinfer's TRT-LLM
    ``allreduce_fusion(pattern=kAllReduce)``.

    The communicator owns its own flashinfer workspace, so the same
    ``GroupCoordinator`` that runs AR+RMSNorm fusion (managed in
    ``flashinfer_comm_fusion.py``) does not have to share buffers with
    standalone AR.

    Lifecycle:
      - ``__init__`` records the group and device but does not allocate
        IPC buffers.
      - ``initialize_workspace`` must run once *before* CUDA graph capture.
        Failure on any rank is broadcast via the cpu_group so all ranks
        agree to disable; a divergent decision would hang graph capture.
      - ``should_trtllm_ar`` is the runtime predicate (token-count, dtype,
        contiguity, hidden_dim match).
      - ``trtllm_all_reduce`` runs the kernel out-of-place and returns the
        output tensor.
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup | None,
        device: int | str | torch.device,
    ) -> None:
        self.disabled = True
        self._workspace_manager = None
        self._flashinfer_comm = None

        if not _is_cuda:
            return

        try:
            from sglang.srt.layers.flashinfer_comm_fusion import (
                FlashInferWorkspaceManager,
                _flashinfer_comm,
                is_flashinfer_allreduce_unavailable,
            )
        except Exception as e:
            logger.debug("TRTLLMAllReduce disabled: flashinfer unavailable (%s)", e)
            return

        if _flashinfer_comm is None or is_flashinfer_allreduce_unavailable():
            logger.debug(
                "TRTLLMAllReduce disabled: flashinfer.comm fusion API unavailable"
            )
            return

        try:
            world_size = dist.get_world_size(group=cpu_group)
            rank = dist.get_rank(group=cpu_group)
        except Exception as e:
            logger.warning("TRTLLMAllReduce disabled: %s", e)
            return

        if world_size <= 1:
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)

        self._flashinfer_comm = _flashinfer_comm
        self._workspace_manager = FlashInferWorkspaceManager()
        self._cpu_group = cpu_group
        self._device_group = device_group
        self._device = device
        self._world_size = world_size
        self._rank = rank
        self._max_token_num: int | None = None
        self._hidden_dim: int | None = None
        self._dtype: torch.dtype | None = None
        # Stays disabled until initialize_workspace succeeds on all ranks.

    def initialize_workspace(
        self,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot: bool | None = None,
    ) -> bool:
        """Allocate the IPC workspace. Must run before CUDA graph capture.

        Returns True iff the workspace is ready on all ranks.
        """
        if self._workspace_manager is None:
            return False
        if dtype not in self._SUPPORTED_DTYPES:
            logger.debug("TRTLLMAllReduce: dtype %s not supported", dtype)
            return self._sync_disable()

        local_ok = False
        try:
            self._workspace_manager.initialize(
                world_size=self._world_size,
                rank=self._rank,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                use_oneshot=use_oneshot,
                device_group=self._device_group,
                cpu_group=self._cpu_group,
            )
            local_ok = bool(self._workspace_manager.initialized)
        except Exception as e:
            logger.warning("TRTLLMAllReduce workspace init failed: %s", e)
            local_ok = False

        # Sync across cpu_group: if any rank failed, all disable.
        try:
            flag = torch.tensor([0 if local_ok else 1], dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self._cpu_group)
            global_ok = local_ok and flag.item() == 0
        except Exception as e:
            logger.warning("TRTLLMAllReduce availability sync failed: %s", e)
            global_ok = False

        if not global_ok:
            try:
                self._workspace_manager.cleanup()
            except Exception:
                pass
            self.disabled = True
            return False

        self._max_token_num = max_token_num
        self._hidden_dim = hidden_dim
        self._dtype = dtype
        self.disabled = False
        logger.info(
            "TRTLLMAllReduce ready (rank=%d world_size=%d hidden_dim=%d "
            "max_token_num=%d dtype=%s)",
            self._rank,
            self._world_size,
            hidden_dim,
            max_token_num,
            dtype,
        )
        return True

    def _sync_disable(self) -> bool:
        try:
            flag = torch.tensor([1], dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self._cpu_group)
        except Exception:
            pass
        self.disabled = True
        return False

    def should_trtllm_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.device.type != "cuda":
            return False
        if inp.dtype != self._dtype:
            return False
        if not inp.is_contiguous():
            return False
        if inp.dim() < 1 or inp.shape[-1] != self._hidden_dim:
            return False
        token_num = inp.numel() // inp.shape[-1]
        if token_num <= 0 or token_num > self._max_token_num:
            return False
        return True

    def trtllm_all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Out-of-place TRT-LLM AR. Caller must have passed should_trtllm_ar."""
        from flashinfer.comm import AllReduceFusionPattern

        token_num = inp.numel() // inp.shape[-1]
        out = torch.empty_like(inp)
        inp_2d = inp.view(token_num, inp.shape[-1])
        out_2d = out.view(token_num, inp.shape[-1])
        self._flashinfer_comm.allreduce_fusion(
            input=inp_2d,
            workspace=self._workspace_manager.workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            output=out_2d,
            fp32_acc=True,
        )
        return out

    def close(self) -> None:
        if self._workspace_manager is not None:
            try:
                self._workspace_manager.cleanup()
            except Exception:
                pass
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
