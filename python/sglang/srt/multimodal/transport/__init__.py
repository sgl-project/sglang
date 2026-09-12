"""GPU transports for multimodal feature tensors."""

from typing import Literal

from sglang.srt.runtime_context import get_parallel

TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


def determine_tensor_transport_mode() -> TensorTransportMode:
    """Select tensor transport from node topology, not rendezvous configuration.

    A rendezvous address may be present for single-node TP. Conversely, Ray can
    inject the address only into scheduler actors for a multi-node deployment,
    and external launchers may use an environment-based rendezvous instead.
    """
    if get_parallel().nnodes > 1:
        # CUDA IPC and POSIX shared memory are local to one node.
        return "default"
    return "cuda_ipc"
