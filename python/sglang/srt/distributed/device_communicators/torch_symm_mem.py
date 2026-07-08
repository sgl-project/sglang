# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/bf214ca22625e311a2c4c0dfbf7af19128f4919c/vllm/distributed/device_communicators/symm_mem.py
import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.all_reduce_utils import (
    TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_cuda, is_hip

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    _is_cuda = is_cuda()
    _is_hip = is_hip()

    torch_symm_mem_available = False
    if _is_cuda:
        torch_symm_mem_available = True
except ImportError:
    torch_symm_mem_available = False


logger = logging.getLogger(__name__)


class TorchSymmMemCommunicator:
    """
    Thin wrapper around torch-symmetric-memory collectives.

    This communicator:
      - Validates device capability and world size.
      - Allocates a shared symmetric buffer.
      - Chooses between 'multimem' and 'two-shot' all-reduce kernels.
      - Exposes a fast-path all_reduce() compatible with bfloat16 inputs.

    If any prerequisite is not met, the instance remains disabled and will
    decline to perform symmetric-memory all-reduce.
    """

    # Mapping: compute capability major -> supported world sizes for multimem
    # If the current (cc_major, world_size) is not listed, we fall back
    # to the two-shot path.
    _WORLD_SIZES_MULTIMEM = {
        9: [4, 6, 8],
        10: [6, 8],
    }

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        """
        Args:
            group: Torch process group used for rendezvous and naming.
            device: Target CUDA device (index, 'cuda:X', or torch.device).
        """

        # disabled: entire communicator unusable.
        # allreduce_disabled: only the allreduce fast path is off; buffers
        # may still serve fused-kernel contexts (RS/AG).
        self.disabled = True
        self.buffer = None
        self.max_size = 0
        self.allreduce_disabled = True
        self.use_cp = False  # set True during CP-mode forward
        # Lazy fused-kernel contexts (cached on first use).
        self._moe_rs_ctx = None
        self._moe_rs_key = None
        self._ag_gemm_ctx = None
        self._ag_gemm_key = None
        self._ag_gemm_stream: Optional[torch.cuda.Stream] = None

        if not torch_symm_mem_available:
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.device_capability = torch.cuda.get_device_capability(device)[0]
        supported_max_sizes = TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES.get(
            self.device_capability
        )
        if supported_max_sizes is None:
            logger.warning(
                "TorchSymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if self.world_size not in supported_max_sizes:
            logger.warning(
                "TorchSymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        self.max_size = supported_max_sizes[self.world_size]
        self.buffer = torch_symm_mem.empty(
            self.max_size // self.dtype.itemsize,
            device=self.device,
            dtype=self.dtype,
        )
        handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        # Enable communicator; allreduce gated separately by multicast.
        self.disabled = False
        if handle.multicast_ptr == 0:
            logger.warning(
                "TorchSymmMemCommunicator: torch symmetric memory "
                "multicast operations are not supported; symm-mem all-reduce "
                "fast path disabled (fused-kernel contexts may still work)."
            )
            self.allreduce_disabled = True
            return
        self.allreduce_disabled = False

    def set_use_cp(self, value: bool) -> None:
        """Set the CP mode flag for fused kernels."""
        self.use_cp = value

    @staticmethod
    def get_active_comm() -> "Optional[TorchSymmMemCommunicator]":
        """Return the TP group's communicator if enabled and use_cp is active, else None."""
        from sglang.srt.distributed import get_tp_group

        # TODO(zxdu): maybe use cp group here?
        comm = get_tp_group().torch_symm_mem_comm
        if comm is None or comm.disabled or not comm.use_cp:
            return None
        return comm

    def should_torch_symm_mem_allreduce(self, inp: torch.Tensor):
        """
        Fast-path eligibility check for a given tensor.

        Conditions:
          - Communicator must be enabled.
          - dtype must be bfloat16 (matches kernel + buffer dtype).
          - Total byte size must be 4-byte aligned (hardware requirement).
          - Payload must be smaller than the symmetric-memory max size.

        Returns:
            True if the symmetric-memory path can handle this tensor.
        """
        if self.disabled or self.allreduce_disabled:
            return False
        if inp.device != self.device:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        # enforce 4-byte alignment
        if inp_size % 4 != 0:
            return False
        return inp_size < self.max_size

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Perform an in-place sum all-reduce via torch symmetric memory.

        Args:
            inp: Input tensor on the target CUDA device (bfloat16).
            out: Optional output tensor; if omitted, a new tensor is allocated.

        Returns:
            The reduced tensor (same shape as inp), or None if disabled.

        Implementation details:
            - Stages 'inp' into the symmetric buffer.
            - Selects 'multimem' or 'two_shot' kernel based on topology.
            - Writes the result into 'out' and returns it.
        """
        if not self.should_torch_symm_mem_allreduce(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[: inp.numel()].copy_(inp.view(-1))
        if self.world_size in self._WORLD_SIZES_MULTIMEM.get(
            self.device_capability, ()
        ):
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out

    def _get_max_forward_tokens(self) -> int:
        """Return max tokens per forward (chunked_prefill_size or max_prefill_tokens)."""
        server_args = get_global_server_args()
        cps = server_args.chunked_prefill_size
        if cps is not None and cps > 0:
            return cps
        return server_args.max_prefill_tokens

    def get_or_create_moe_rs_ctx(
        self,
        N: int,
        num_experts: int,
        topk: int,
        dtype: torch.dtype,
        n_chunks_max: int = 8,
    ):
        """Lazy-init / cache the MoE reduce-scatter symm-mem context."""
        if self.disabled:
            return None
        key = (N, num_experts, topk, dtype, n_chunks_max, self.world_size)
        if self._moe_rs_key == key and self._moe_rs_ctx is not None:
            return self._moe_rs_ctx
        if self._moe_rs_ctx is not None:
            try:
                self._moe_rs_ctx.finalize()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "TorchSymmMemCommunicator: failed to finalize stale MoE RS "
                    "context: %s",
                    e,
                )
            self._moe_rs_ctx = None
            self._moe_rs_key = None

        from sglang.srt.distributed.device_communicators.symm_mem_kernels import (
            create_moe_rs_symm_mem_context,
        )

        max_M = self._get_max_forward_tokens()

        self._moe_rs_ctx = create_moe_rs_symm_mem_context(
            # TODO(zxdu): check whether we should use cp group here or not?
            rank=dist.get_rank(self.group),
            world_size=self.world_size,
            local_world_size=self.world_size,
            max_token_num=max_M,
            hidden_dim=N,
            num_experts=num_experts,
            topk=topk,
            input_dtype=dtype,
            n_chunks_max=n_chunks_max,
            group=self.group,
        )
        self._moe_rs_key = key
        return self._moe_rs_ctx

    def get_or_create_ag_gemm_ctx(
        self,
        K: int,
        NUM_COMM_SMS: int = 0,
    ):
        """Lazy-init / cache the all-gather + GEMM symm-mem context."""
        if self.disabled:
            return None
        key = (K, NUM_COMM_SMS, self.world_size)
        if self._ag_gemm_key == key and self._ag_gemm_ctx is not None:
            return self._ag_gemm_ctx
        if self._ag_gemm_ctx is not None:
            try:
                self._ag_gemm_ctx.finalize()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "TorchSymmMemCommunicator: failed to finalize stale AG+GEMM "
                    "context: %s",
                    e,
                )
            self._ag_gemm_ctx = None
            self._ag_gemm_key = None

        from sglang.srt.distributed.device_communicators.symm_mem_kernels import (
            create_allgather_gemm_context_symm_mem,
        )

        if self._ag_gemm_stream is None:
            self._ag_gemm_stream = torch.cuda.Stream(
                device=self.device, priority=-1
            )

        max_M = self._get_max_forward_tokens()

        self._ag_gemm_ctx = create_allgather_gemm_context_symm_mem(
            ag_stream=self._ag_gemm_stream,
            rank=dist.get_rank(self.group),
            world_size=self.world_size,
            max_M=max_M // self.world_size,
            K=K,
            NUM_COMM_SMS=NUM_COMM_SMS,
            enable_multicast=False,
            group=self.group,
        )
        self._ag_gemm_key = key
        return self._ag_gemm_ctx


