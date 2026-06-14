"""Overlap the MoE shared-expert add with the down-LoRA shrink (gemm A).

In the Qwen dual-stream decode path (``Qwen2MoeSparseMoeBlock.forward_normal_dual_stream``)
the shared expert runs on the main stream while the routed experts + trtllm LoRA
MoE run on the alt stream; ``final_hidden_states += shared_output`` then runs on
the main stream only after the WHOLE alt-stream chain (base MoE -> finalize ->
down-LoRA shrink -> down-LoRA expand) joins — putting a ~2us elementwise add at
the tail of every MoE layer's critical path.

The add's real dependency is only the base-MoE finalize (the trtllm op with
``do_finalize=True`` writing the output buffer): the down-LoRA shrink writes a
separate intermediate, and the down-LoRA expand atomic-adds into the same output
buffer, so addition order is commutative — the only hard constraint is that the
non-atomic shared add and the expand must not run CONCURRENTLY on the buffer.
(The cross-rank ``tensor_model_parallel_all_reduce`` happens after all of this
at the model layer, unchanged.)

Protocol (gated by ``SGLANG_OPT_LORA_SHARED_ADD_OVERLAP``):

1. The model layer stages ``(shared_output, producer_stream)`` via
   :func:`stage_shared_expert_add` after computing the shared expert and before
   forking the routed experts to the alt stream.
2. The LoRA dispatch, right after the trtllm op returns (finalize done), calls
   :func:`maybe_overlap_staged_shared_add(output)`: it records ``base_ready`` on
   the current (alt) stream, enqueues ``wait(base_ready); output += shared;
   record(add_done)`` on the producer (main) stream — main-stream program order
   already guarantees ``shared_output`` is ready there — and returns ``add_done``.
3. The down-LoRA ``merged_experts_fused_moe_lora_add`` waits on ``add_done``
   right before launching the expand kernel, so the shrink (+ stage-B routing)
   overlaps the add and the expand never races it.
4. After the dual-stream join the model layer calls
   :func:`unstage_shared_expert_add`; if the dispatch never consumed the staging
   (prefill / non-virtual-store / fallback paths) it gets the tensor back and
   performs the original add itself — byte-identical fallback behavior.

The state is a single slot: MoE layers run sequentially within one scheduler
process, and the stage/consume pair lives within a single layer forward.
"""

from typing import Optional, Tuple

import torch

from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

_PENDING: Optional[Tuple[torch.Tensor, torch.cuda.Stream]] = None

# Keep events recorded during cuda-graph capture alive so the captured
# cross-stream waits aren't torn down before graph instantiation (same pattern
# as moe_overlap._LORA_OVERLAP_EVENTS). Eager runs rely on deferred destroy.
_SHARED_ADD_EVENTS: list = []


def shared_add_overlap_enabled() -> bool:
    return lora_envs.SGLANG_OPT_LORA_SHARED_ADD_OVERLAP.get()


def stage_shared_expert_add(
    shared_output: torch.Tensor, producer_stream: torch.cuda.Stream
) -> None:
    """Stage the shared-expert output for the LoRA dispatch to add.

    ``producer_stream`` is the stream ``shared_output`` was computed on (the
    main stream); the overlapped add is enqueued there so its data dependency
    on the shared expert is carried by stream program order.
    """
    global _PENDING
    _PENDING = (shared_output, producer_stream)


def unstage_shared_expert_add() -> Optional[torch.Tensor]:
    """Reclaim a staged-but-unconsumed shared add (fallback paths).

    Returns the staged tensor if the dispatch did NOT consume it (the model
    layer must then do the add itself), or None if it was consumed (the add is
    already enqueued on the producer stream).
    """
    global _PENDING
    if _PENDING is None:
        return None
    shared_output, _ = _PENDING
    _PENDING = None
    return shared_output


def maybe_overlap_staged_shared_add(output: torch.Tensor) -> Optional[torch.cuda.Event]:
    """Enqueue the staged shared-expert add overlapped with the down-LoRA shrink.

    Call from the LoRA dispatch right after the base-MoE finalize has been
    enqueued on the current stream with ``output`` fully written. Returns the
    ``add_done`` event the down-LoRA expand must wait on before atomic-adding
    into ``output``, or None when nothing was staged.
    """
    global _PENDING
    if _PENDING is None:
        return None
    shared_output, producer_stream = _PENDING
    current_stream = torch.cuda.current_stream()
    if producer_stream == current_stream:
        # Single-stream caller: nothing to overlap. Leave the staging in place
        # so the model layer reclaims it and does the add as before.
        return None
    _PENDING = None

    base_ready = torch.cuda.Event()
    base_ready.record(current_stream)
    add_done = torch.cuda.Event()
    with torch.cuda.stream(producer_stream):
        producer_stream.wait_event(base_ready)
        output.add_(shared_output)
        add_done.record(producer_stream)

    if torch.cuda.is_current_stream_capturing():
        _SHARED_ADD_EVENTS.append(base_ready)
        _SHARED_ADD_EVENTS.append(add_done)
    return add_done
