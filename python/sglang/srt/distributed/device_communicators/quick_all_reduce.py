# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import os
from enum import Enum
from functools import cache
from typing import Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_full_nvlink,
    is_weak_contiguous,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.utils import is_cuda, is_hip

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


def _new_vmm_pool_name(
    group: ProcessGroup, ranks_tag: str, device: torch.device
) -> str:
    """Create a communicator generation that cannot reuse stale Store keys."""
    group_ranks = dist.get_process_group_ranks(group)
    if not group_ranks:
        raise RuntimeError("QuickAllReduce VMM process group has no ranks")
    source_rank = group_ranks[0]
    nonce = 0
    if dist.get_rank() == source_rank:
        nonce = int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)
    # The QuickReduce process group uses RCCL, so its collectives must use a
    # device tensor. A CPU tensor here fails before VMM fd exchange begins.
    nonce_tensor = torch.tensor([nonce], dtype=torch.int64, device=device)
    dist.broadcast(nonce_tensor, src=source_rank, group=group)
    generation = hashlib.blake2s(
        int(nonce_tensor.item()).to_bytes(8, "little"), digest_size=8
    ).hexdigest()
    return f"sglang_quickreduce_{ranks_tag}_g{generation}"


def _raise_if_any_vmm_phase_failed(
    group: ProcessGroup, phase: str, local_error: Exception | None
) -> None:
    """Make VMM setup succeed or fail consistently on every rank."""
    local_status = None
    if local_error is not None:
        local_status = f"{type(local_error).__name__}: {local_error}"
    statuses = [None] * dist.get_world_size(group)
    dist.all_gather_object(statuses, local_status, group=group)
    failures = [
        f"rank {rank}: {status}" for rank, status in enumerate(statuses) if status
    ]
    if failures:
        raise RuntimeError(
            f"QuickAllReduce VMM {phase} failed across ranks: " + "; ".join(failures)
        )


def _select_quick_reduce_ipc_backend(device: torch.device) -> str:
    ipc_backend = os.environ.get("ROCM_QUICK_REDUCE_IPC_BACKEND", "auto")
    ipc_backend = ipc_backend.strip().lower()
    if ipc_backend not in ("auto", "hipipc", "vmm"):
        raise ValueError(f"Unsupported ROCM_QUICK_REDUCE_IPC_BACKEND={ipc_backend!r}")
    if ipc_backend != "auto":
        return ipc_backend

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    props = torch.cuda.get_device_properties(device_index)
    arch = getattr(props, "gcnArchName", "")
    return "vmm" if "gfx950" in arch else "hipipc"


class QuickAllReduce:

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
        self.disabled = True
        self._ptr = 0
        if not qr_rocm_arch_available():
            logger.debug(
                "Custom quick allreduce is only supported on ROCm MI300 series."
            )
            return

        if not ops.IS_QUICK_AR_AVAILABLE:
            # disable because of missing quick reduce library
            # e.g. in a cuda environment
            logger.info(
                "Custom quick allreduce is disabled because "
                "of missing custom quick allreduce library"
            )
            return

        self.group = group
        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "Custom quick allreduce should be attached to a non-NCCL group."
        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom quick allreduce for
            # multi-node case.
            logger.warning(
                "Custom quick allreduce is disabled because this "
                "process group spans across nodes."
            )
            return
        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        self.rank = rank
        self.world_size = world_size
        if world_size == 1:
            # No need to initialize QuickReduce for single GPU case.
            return

        if world_size not in QuickAllReduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom quick allreduce is disabled due to an "
                "unsupported world size: %d. Supported world sizes: %s.",
                world_size,
                str(QuickAllReduce._SUPPORTED_WORLD_SIZES),
            )
            return

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
            logger.debug(
                "Custom quick allreduce is disabled because it's not supported "
                "on more than two PCIe-only GPUs. "
            )
            return

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
            logger.warning(
                "Custom quick allreduce:",
                f"Invalid quantization level: {regime_str}. "
                "Supported levels: "
                f"{list(QuickReduceRegime.__members__.keys())}",
            )
            return

        if regime_str == "NONE":
            logger.debug(
                "Custom quick allreduce is disabled based "
                "on env variable "
                "ROCM_QUICK_REDUCE_QUANTIZATION='NONE'"
            )
            return
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
        ipc_backend = _select_quick_reduce_ipc_backend(self.device)
        self._uses_vmm = ipc_backend == "vmm"
        if self._uses_vmm:
            self._init_vmm(qr_max_size)
        else:
            self._ptr = ops.init_custom_qr(self.rank, self.world_size, qr_max_size)
            self.create_shared_buffer()
        self.qr_max_size = qr_max_size if qr_max_size > 0 else ops.qr_max_size()
        self.disabled = False

    def _init_vmm(self, qr_max_size: int) -> None:
        from sglang.srt.distributed.device_communicators.quick_all_reduce_vmm import (
            exchange_vmm_fds,
        )

        store = dist.distributed_c10d._get_default_store()
        ranks_tag = "_".join(map(str, sorted(dist.get_process_group_ranks(self.group))))
        uncached = bool(int(os.environ.get("ROCM_QUICK_REDUCE_VMM_UNCACHED", "1")))

        local_fd = -1
        local_size = 0
        peer_fds = []
        arch = ""

        def cleanup_ptr() -> None:
            if not self._ptr:
                return
            try:
                ops.qr_destroy(self._ptr)
            except Exception:
                logger.exception("Failed to clean up QuickAllReduce VMM state")
            finally:
                self._ptr = 0

        try:
            allocation_error = None
            try:
                device_index = (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
                props = torch.cuda.get_device_properties(device_index)
                arch = getattr(props, "gcnArchName", "")
                if "gfx950" not in arch:
                    raise RuntimeError(
                        "QuickAllReduce VMM IPC is only supported on gfx950, "
                        f"got {arch!r}"
                    )
                self._ptr, local_fd, local_size = ops.init_custom_qr_vmm(
                    self.rank,
                    self.world_size,
                    device_index,
                    qr_max_size,
                    uncached,
                )
            except Exception as exc:
                allocation_error = exc
            _raise_if_any_vmm_phase_failed(
                self.group, "local allocation", allocation_error
            )

            pool_name = _new_vmm_pool_name(self.group, ranks_tag, self.device)
            exchange_error = None
            try:
                peer_fds, peer_sizes = exchange_vmm_fds(
                    self.rank,
                    self.world_size,
                    pool_name,
                    local_fd,
                    local_size,
                    store,
                    ranks_tag,
                )
            except Exception as exc:
                exchange_error = exc
                peer_sizes = []
            _raise_if_any_vmm_phase_failed(
                self.group, "file descriptor exchange", exchange_error
            )

            open_error = None
            try:
                ops.qr_open_vmm_handles(self._ptr, peer_fds, peer_sizes)
            except Exception as exc:
                open_error = exc
            # This consensus is also the readiness barrier: no rank can enter a
            # kernel until every peer has mapped and initialized its buffers.
            _raise_if_any_vmm_phase_failed(self.group, "peer mapping", open_error)
        except Exception:
            cleanup_ptr()
            raise
        finally:
            if local_fd >= 0:
                os.close(local_fd)
            for fd in peer_fds:
                if fd >= 0:
                    os.close(fd)

        logger.info(
            "QuickAllReduce selected VMM IPC for rank %d/%d on %s (uncached=%s)",
            self.rank,
            self.world_size,
            arch,
            uncached,
        )

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

    def should_quick_allreduce(self, inp: torch.Tensor):
        """
        Check if quickreduce is available
        """
        if self.disabled:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom quick allreduce requires input byte size to be
        # multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        dtype = inp.dtype
        if self.use_fp16_kernels:
            dtype = torch.float16
        return (
            inp_size <= self.qr_max_size
            and inp_size
            >= self._QR_MIN_SIZE[(dtype, self.world_size)][self.qr_quant_level.value]
        )

    def quick_all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        """Performs an out-of-place custom quick all reduce."""
        # quick allreduce doesn't require a separate graph mode,
        # as QR uses static IPC buffer.
        if out is None:
            out = torch.empty_like(inp)
        ops.qr_all_reduce(
            self._ptr, inp, out, self.qr_quant_level.value, self.use_fp16_kernels
        )
        return out

    def close(self):
        if getattr(self, "_ptr", None):
            ptr = self._ptr
            # qr_destroy always deletes the native object, even when a later
            # HIP cleanup step reports an error. Clear Python ownership first
            # so finalization cannot call native destroy on a stale pointer.
            self._ptr = 0
            self.disabled = True
            if ops is not None:
                ops.qr_destroy(ptr)
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors run during interpreter and worker teardown. Explicit
            # close() still reports cleanup errors to callers, while finalizer
            # cleanup must never surface an unraisable exception.
            pass
