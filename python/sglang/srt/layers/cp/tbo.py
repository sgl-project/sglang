"""Asynchronous interleave gathers shared by the two CP prefill microbatches."""

from dataclasses import dataclass

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    _tbo_event,
    get_dp_tbo_comm_stream,
    get_tbo_persistent_tensor,
    is_allocation_symmetric,
)
from sglang.srt.runtime_context import get_parallel


@dataclass
class CpTboAllGatherHandle:
    """An in-flight round-robin CP all-gather used by decomposed TBO attention."""

    input_tensor: torch.Tensor
    output_tensor: torch.Tensor
    reranged_tensor: torch.Tensor
    event: torch.cuda.Event
    cp_size: int


def cp_tbo_all_gather_rerange_output_async(
    input_tensor: torch.Tensor,
    cp_size: int,
    forward_batch,
    *,
    kind: str,
) -> CpTboAllGatherHandle:
    """Launch one round-robin CP all-gather without blocking the compute stream.

    All attention and MoE-boundary collectives intentionally keep using the
    shared TBO communication stream. ``kind`` gives each simultaneously
    outstanding attention gather its own persistent buffer and event while
    still reusing those resources across layers and forwards.
    """
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_interleave,
    )

    assert is_dsa_prefill_cp_interleave()
    assert input_tensor.ndim == 2
    subbatch = getattr(forward_batch, "tbo_cp_subbatch_index", None)
    assert subbatch is not None

    output_shape = (input_tensor.shape[0] * cp_size, input_tensor.shape[1])
    with use_symmetric_memory(
        get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        output_tensor = get_tbo_persistent_tensor(
            ("cp_attn_ag", kind, subbatch),
            output_shape,
            input_tensor.dtype,
            input_tensor.device,
        )
    reranged_tensor = get_tbo_persistent_tensor(
        ("cp_attn_rerange", kind, subbatch),
        output_shape,
        input_tensor.dtype,
        input_tensor.device,
    )
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    event = _tbo_event(("cp_attn_gather", kind, subbatch))

    # Keep the temporary input valid until the communication stream has read it.
    input_tensor.record_stream(comm)
    output_tensor.record_stream(comm)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        get_parallel().attn_cp_group.cp_all_gather_into_tensor_async(
            output_tensor,
            input_tensor,
            comm,
        )
        event.record(comm)

    return CpTboAllGatherHandle(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        reranged_tensor=reranged_tensor,
        event=event,
        cp_size=cp_size,
    )


def cp_tbo_all_gather_rerange_output_wait(
    handle: CpTboAllGatherHandle,
) -> torch.Tensor:
    """Wait for and interleave one round-robin CP all-gather result."""
    torch.cuda.current_stream().wait_event(handle.event)
    output_tensor = handle.output_tensor
    out_shape = output_tensor.shape
    gathered_by_rank = output_tensor.view(handle.cp_size, -1, *out_shape[1:]).transpose(
        0, 1
    )
    handle.reranged_tensor.view_as(gathered_by_rank).copy_(gathered_by_rank)
    return handle.reranged_tensor
