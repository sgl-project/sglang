"""Device ordering prerequisites for concurrent PDMux communicators."""

import logging
import os

import torch

from sglang.srt.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from sglang.srt.utils.common import get_cuda_driver_bindings, get_cuda_version

logger = logging.getLogger(__name__)


def configure_pdmux_nccl_launch_order() -> None:
    """Run before NCCL initialization, without draining either model lane.

    Host launch order alone does not order independent NCCL communicators on
    the device. NCCL 2.26+ implicit ordering supplies that dependency; CUDA
    runtime and driver 12.3+ permit overlapping communication kernels. The
    scheduler must still submit matching operations in the same host order.
    """
    if torch.version.cuda is None:
        return
    setting = os.environ.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    if setting != "1":
        raise RuntimeError(
            "Concurrent PDMux communicators require NCCL_LAUNCH_ORDER_IMPLICIT=1; "
            f"the environment explicitly sets it to {setting!r}."
        )
    torch_nccl = torch.cuda.nccl.version()
    library = NCCLLibrary()
    pynccl = library.ncclGetRawVersion()
    driver_error, driver_version = get_cuda_driver_bindings().cuDriverGetVersion()
    if (
        torch_nccl < (2, 26, 0)
        or pynccl < 22600
        or get_cuda_version() < (12, 3)
        or int(driver_error) != 0
        or driver_version < 12030
    ):
        raise RuntimeError(
            "PDMux communicator overlap requires NCCL >= 2.26 in both PyTorch "
            "and PyNCCL, and CUDA runtime/driver >= 12.3. "
            f"Found PyTorch NCCL {torch_nccl}, PyNCCL {pynccl}, "
            f"CUDA {torch.version.cuda}, driver {driver_version}."
        )
    logger.info(
        "PDMux launch ordering: implicit=1, PyTorch NCCL=%s, PyNCCL=%s (%s), "
        "CUDA=%s, driver=%s, NCCL_GRAPH_MIXING_SUPPORT=%s",
        torch_nccl,
        pynccl,
        getattr(getattr(library, "lib", None), "_name", "unknown"),
        torch.version.cuda,
        driver_version,
        os.environ.get("NCCL_GRAPH_MIXING_SUPPORT", "unset (NCCL default)"),
    )
