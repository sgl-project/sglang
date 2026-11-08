"""CUDA-IPC synchronization and stream-ordered peer copies."""

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, load_jit


@cache_once
def load_ipc_a2a_sync():
    # the shared jit loader releases its lock on process death and publishes
    # complete builds atomically, unlike load_inline's persistent lock file
    return load_jit(
        "ipc_a2a_sync",
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
        cuda_files=["distributed/ipc_a2a.cuh"],
        cuda_wrappers=[
            ("spin_wait", "ipc_a2a::spin_wait"),
            ("bump_signal", "ipc_a2a::bump_signal"),
            ("copy_peer_strided", "ipc_a2a::copy_peer_strided"),
        ],
    )
