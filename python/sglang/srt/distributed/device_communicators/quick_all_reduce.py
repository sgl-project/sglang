# SPDX-License-Identifier: Apache-2.0

import logging
import os
from enum import Enum
from functools import cache
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.utils import is_cuda, is_hip

from .base import AllReduceMode, BaseCommunicator
from .custom_all_reduce_utils import is_full_nvlink, is_weak_contiguous

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()


@cache
def qr_rocm_arch_available():
    if not _is_hip:
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        supported_archs = ["gfx94", "gfx95"]
        return any(gfx in gcn_arch for gfx in supported_archs)
    except Exception as e:
        logger.warning("Failed to determine ROCm for quick allreduce: %s", e)
        return False


class QuickReduceRegime(Enum):
    FP = 0
    INT8 = 1
    INT6 = 2
    INT4 = 3
    NONE = 4


MB = 1024 * 1024


class QuickAllReduce(BaseCommunicator):
    name = "quick_all_reduce"

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    # The following data is based on kernel tests.
    # In this order [FP, INT8, INT6, INT4].
    _QR_MIN_SIZE = {
        (torch.float16, 2): [1 * MB, 2 * MB, 2 * MB, 1 * MB],
        (torch.float16, 4): [1 * MB, 16 * MB, 4 * MB, 2 * MB],
        (torch.float16, 8): [16 * MB, 4 * MB, 4 * MB, 2 * MB],
        (torch.bfloat16, 2): [2 * MB, 8 * MB, 8 * MB, 8 * MB],
        (torch.bfloat16, 4): [8 * MB, 64 * MB, 64 * MB, 16 * MB],
        (torch.bfloat16, 8): [16 * MB, 2048 * MB, 2048 * MB, 2048 * MB],
    }

    def __init__(
        self, group: ProcessGroup, device: Union[int, str, torch.device]
    ) -> None:
        """
        Custom allreduce provides non-destructive acceleration and is
        available for CUDA and ROCm MI300 series.
        Custom quick allreduce leverages quantization for further
        acceleration on ROCm. It currently supports Q8, Q6, and Q4
        quantization formats and FP(float16, bfloat16).
        Quick allreduce is designed as a complement to custom allreduce.
        Its initialization requires even stricter conditions.
        Only the ROCm MI300 series is supported for quick allreduce at
        this time.
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        # Set unconditionally so that close() (reached via __del__ even when
        # __init__ raises below) can read them without defensive getattr.
        self._disabled = True
        self._ptr = 0

        if not qr_rocm_arch_available():
            raise RuntimeError(
                "quick all-reduce is only supported on ROCm MI300 series"
            )

        if not ops.IS_QUICK_AR_AVAILABLE:
            raise RuntimeError("quick all-reduce library is not available")

        self.group = group
        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "Custom quick allreduce should be attached to a non-NCCL group."
        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom quick allreduce for
            # multi-node case.
            raise RuntimeError("quick all-reduce requires all ranks to be on one node")
        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        self.rank = rank
        self.world_size = world_size
        assert world_size > 1
        super().__init__(world_size=world_size)

        if world_size not in QuickAllReduce._SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "unsupported quick all-reduce world size: "
                f"{world_size}; supported sizes: {QuickAllReduce._SUPPORTED_WORLD_SIZES}"
            )

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(torch.cuda.device_count()))
        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu")
            for _ in range(self.world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom quick allreduce is not supported
        # this checks hardware and driver support for NVLink
        if _is_cuda or _is_hip:
            self.fully_connected = is_full_nvlink(physical_device_ids, self.world_size)
        if self.world_size > 2 and not self.fully_connected:
            raise RuntimeError(
                "quick all-reduce does not support more than two PCIe-only GPUs"
            )

        self.init_quick_all_reduce()

    def init_quick_all_reduce(self):
        # On RocM, bfloat16 kernels are slower than fp16
        # due to slower match operations
        # If environment variable is set to 1, we convert input to fp16
        self.use_fp16_kernels = int(
            os.environ.get("ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", 1)
        )
        regime_str = os.environ.get("ROCM_QUICK_REDUCE_QUANTIZATION", "NONE")
        if regime_str not in QuickReduceRegime.__members__:
            raise ValueError(
                "invalid quick all-reduce quantization level: "
                f"{regime_str}; supported levels: {list(QuickReduceRegime.__members__.keys())}"
            )

        if regime_str == "NONE":
            raise RuntimeError(
                "quick all-reduce is disabled by ROCM_QUICK_REDUCE_QUANTIZATION='NONE'"
            )
        self.qr_quant_level = QuickReduceRegime[regime_str]

        # TODO: If the dtype is not bfloat16 or then float16,
        # quickallreduce should not be created.

        # ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB is specified in MB
        qr_max_size = int(os.environ.get("ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", 0))
        if qr_max_size > 0:
            if qr_max_size < 1:
                logger.info(
                    "You should not set a max_size smaller than 1MB, which can "
                    "lead to error or degradation to custom allreduce or rccl."
                )
            qr_max_size = qr_max_size * MB
        # If qr_max_size is None, then 2GB is used by default.
        self._ptr = ops.init_custom_qr(self.rank, self.world_size, qr_max_size)
        self.qr_max_size = qr_max_size if qr_max_size > 0 else ops.qr_max_size()
        self.create_shared_buffer()

    def create_shared_buffer(self):
        """
        Creates a shared buffer for quickreduce.
        Has to be called after init_custom_qr
        """
        handle = ops.qr_get_handle(self._ptr)
        world_size = dist.get_world_size(group=self.group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=self.group)
        ops.qr_open_handles(self._ptr, handles)

    def should_use_custom_op(self) -> bool:
        return True

    def get_all_reduce_mode(self, input_: torch.Tensor) -> Optional[AllReduceMode]:
        if self.disabled:
            return None
        if input_.dtype not in self._SUPPORTED_DTYPES:
            return None
        inp_size = input_.numel() * input_.element_size()
        # custom quick allreduce requires input byte size to be
        # multiples of 16
        if inp_size % 16 != 0:
            return None
        if not is_weak_contiguous(input_):
            return None
        dtype = input_.dtype
        if self.use_fp16_kernels:
            dtype = torch.float16
        enabled = (
            inp_size <= self.qr_max_size
            and inp_size
            >= self._QR_MIN_SIZE[(dtype, self.world_size)][self.qr_quant_level.value]
        )
        return AllReduceMode.OUTPLACE if enabled else None

    @BaseCommunicator.validate
    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        """Perform an out-of-place quick all-reduce."""
        # quick allreduce doesn't require a separate graph mode,
        # as QR uses static IPC buffer.
        self.assert_outplace("all_reduce", inplace)
        out = torch.empty_like(input_)
        ops.qr_all_reduce(
            self._ptr, input_, out, self.qr_quant_level.value, self.use_fp16_kernels
        )
        return out

    def close(self):
        if not self._disabled and self._ptr:
            if ops is not None:
                ops.qr_destroy(self._ptr)
            self._ptr = 0
            self._disabled = True

    def __del__(self):
        self.close()
