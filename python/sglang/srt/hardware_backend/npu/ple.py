"""PLE host-table lookup for NPU, which has no UVA view of host memory.

torch_npu maps pinned host memory into the device address space only under
PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True (aclrtHostRegisterV2), and the
Triton gather the CUDA path uses would still need a device-dereferenceable
pointer. Rows are selected on the CPU and staged instead, mirroring the
explicit-copy fallback in vllm-ascend's UvaBufferWrapper.

Under decode graph capture the lookup becomes a break between graph segments
(NPUHostOffloadGraph), and every replay reruns it with that step's IDs.
"""

from functools import partial
from typing import Optional

import torch

from sglang.srt.hardware_backend.npu.graph_runner.host_offload_graph import (
    get_host_offload_graph,
)


def gather_ple_host_rows(
    weight: torch.Tensor,
    flat_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    file_prefetcher: Optional[object] = None,
) -> torch.Tensor:
    """Select this rank's rows on the host and stage them to the NPU as bf16."""
    graph = get_host_offload_graph()
    if graph is not None:
        graph.add_host_call(
            partial(
                gather_ple_host_rows,
                weight,
                flat_ids,
                output,
                vocab_start=vocab_start,
                vocab_end=vocab_end,
                file_prefetcher=file_prefetcher,
            )
        )
        return output
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError(
            "NPU PLE host lookup requires NPUHostOffloadGraph for capture; "
            "use NPUCudaGraphBackend to capture graph segments around the lookup."
        )
    # Blocking D2H: the row set is only known once the IDs are on the host.
    # Normalize on the host: casting int32 IDs on NPU adds a device operation
    # even though all index arithmetic below runs on the CPU.
    ids = flat_ids.to("cpu").long()
    if file_prefetcher is not None:
        file_prefetcher.enqueue(ids, vocab_start=vocab_start, vocab_end=vocab_end)
    in_range = (ids >= vocab_start) & (ids < vocab_end)
    local_ids = torch.where(in_range, ids - vocab_start, 0)
    # index_select always allocates, so masked_fill_ below never touches the table.
    rows = weight.index_select(0, local_ids).to(torch.bfloat16)
    rows.masked_fill_(~in_range[:, None], 0)
    # An unpinned source makes torch_npu synchronize the stream instead of
    # honoring non_blocking (CachingHostAllocator.cpp, process_non_blocking_copy).
    staging = rows.reshape(output.shape).pin_memory()
    output.copy_(staging, non_blocking=True)
    return output
