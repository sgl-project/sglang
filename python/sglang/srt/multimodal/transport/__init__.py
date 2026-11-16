"""GPU transports for multimodal feature tensors."""

from typing import Literal

from sglang.srt.runtime_context import get_device, get_parallel

TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


def determine_tensor_transport_mode() -> TensorTransportMode:
    """Select tensor transport from node topology, not rendezvous configuration.

    A rendezvous address may be present for single-node TP. Conversely, Ray can
    inject the address only into scheduler actors for a multi-node deployment,
    and external launchers may use an environment-based rendezvous instead.
    """
    if not _tokenizer_consumers_on_first_node():
        # CUDA IPC and POSIX shared memory are local to one node.
        return "default"
    return "cuda_ipc"


def _tokenizer_consumers_on_first_node() -> bool:
    """Whether every scheduler that reads tokenizer output shares node 0 with it.

    Only the first pipeline stage's TP group reads tokenizer output, and it
    stays on node 0 unless a TP group spans nodes; later stages receive
    requests from that group, never from the tokenizer.
    """
    parallel = get_parallel()
    if parallel.nnodes == 1:
        return True
    # Ray placement groups can put any rank on any node, and data-parallel
    # replicas spread across nodes.
    return (
        not get_device().use_ray
        and parallel.dp_size == 1
        and parallel.nnodes <= parallel.pp_size
    )
