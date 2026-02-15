import torch
import triton

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout,
        allocate_tensor_memory,
        fence_async_shared,
        get_tmem_reg_layout,
        mbarrier,
        tcgen05_commit,
        tcgen05_mma,
        tma,
    )
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
except ImportError as e:
    raise ImportError(
        f">>> Failed to import Gluon in current triton version {triton.__version__} and "
        f">>> Platform {torch.cuda.get_device_capability()}.\n"
        f">>> Gluon/Blackwell features require: \n"
        f">>> 1. Triton >= 3.6.0 \n"
        f">>> 2. NVIDIA GPU (compute capability == 10.0)\n"
        f">>> 3. Pytorch >= 2.9.0 \n"
        f">>> Error: {e}\n"
        f">>> Set FLA_USE_GLUON=0 to disable and continue."
    ) from e

__all__ = [
    "gluon",
    "gl",
    "TensorMemoryLayout",
    "allocate_tensor_memory",
    "fence_async_shared",
    "get_tmem_reg_layout",
    "mbarrier",
    "tcgen05_commit",
    "tcgen05_mma",
    "tma",
    "TensorDescriptor",
]
