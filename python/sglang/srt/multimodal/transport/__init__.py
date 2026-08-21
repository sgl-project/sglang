"""GPU transports for multimodal feature tensors."""

from typing import TYPE_CHECKING, Literal

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


def determine_tensor_transport_mode(
    server_args: "ServerArgs",
) -> TensorTransportMode:
    """Select tensor transport from node topology, not rendezvous configuration.

    A rendezvous address may be present for single-node TP. Conversely, Ray can
    inject the address only into scheduler actors for a multi-node deployment,
    and external launchers may use an environment-based rendezvous instead.
    """
    # Keep server_args in the signature for existing callers, but read the
    # resolved topology from the published parallel config bag.
    if get_parallel().nnodes > 1:
        # CUDA IPC and POSIX shared memory are local to one node.
        return "default"
    return "cuda_ipc"
