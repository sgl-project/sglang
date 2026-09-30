"""CUDA-IPC sequence-counter kernels for two-rank Ulysses."""

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def load_ipc_a2a_sync():
    # the shared jit loader releases its lock on process death and publishes
    # complete builds atomically, unlike load_inline's persistent lock file
    return load_jit(
        "ipc_a2a_sync",
        cuda_files=["distributed/ipc_a2a.cuh"],
        cuda_wrappers=[
            ("spin_wait", "ipc_a2a::spin_wait"),
            ("bump_signal", "ipc_a2a::bump_signal"),
        ],
    )
