"""Quiesced same-capacity BF16 expert row replacement over the normal TP RPC group."""

from __future__ import annotations

import logging

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_swap import plan_expert_swap
from sglang.srt.eplb.expert_swap_weights import node_staging_bytes, prepare_expert_rows
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule
from sglang.srt.utils.common import get_free_cpu_memory

logger = logging.getLogger(__name__)


def _metadata_layout(metadata):
    tensors = {
        name: value
        for name, value in vars(metadata).items()
        if isinstance(value, torch.Tensor)
    }
    for algorithm, values in metadata._mega_gate_maps.items():
        tensors.update(
            {f"mega.{algorithm}.{index}": value for index, value in enumerate(values)}
        )
    return {
        name: (id(value), value.data_ptr(), tuple(value.shape), value.dtype)
        for name, value in tensors.items()
    }


def _prepare_context(
    manager, requested, expected, max_node_staging_bytes, memory_reserve_bytes
):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    cfg, parallel = get_exec().moe, get_parallel()
    runner = manager.tp_worker.model_runner
    if manager.hot_swap_failed:
        raise RuntimeError(
            "HOT_SWAP_UNSAFE: generation is already held after an unsafe expert update"
        )
    if manager._session is not None:
        raise RuntimeError(
            "Hot replacement is unavailable after an unsafe failure or during a weight-update session"
        )
    if not manager.is_fully_idle() or manager.scheduler._engine_paused:
        raise RuntimeError("Hot replacement requires drained, unpaused generation")
    if (
        parallel.tp_size not in (1, 4)
        or parallel.moe_ep_size != 1
        or parallel.moe_dp_size != 1
        or parallel.attn_dp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.dp_size != 1
        or parallel.pp_size != 1
        or parallel.nnodes != 1
    ):
        raise ValueError(
            "Hot expert replacement requires one-node TP1/TP4 with EP1, DP1 and CP1"
        )
    if (
        cfg.moe_expert_profile is None
        or cfg.moe_runner_backend != "triton"
        or cfg.enable_eplb
        or cfg.enable_paged_experts
        or cfg.ep_num_redundant_experts
        or cfg.expert_distribution_recorder_mode is not None
    ):
        raise ValueError(
            "Hot replacement requires startup physical pruning with BF16 Triton and no EPLB, paging or recorder"
        )
    if (
        not get_schedule().disable_overlap_schedule
        or manager.draft_worker is not None
        or not manager.scheduler.server_args.disable_radix_cache
    ):
        raise ValueError(
            "Initial hot replacement requires non-overlapped generation without prefix caching or a draft"
        )
    if (
        manager.scheduler.enable_hierarchical_cache
        or manager.scheduler.disaggregation_mode != DisaggregationMode.NULL
    ):
        raise ValueError(
            "Initial hot replacement excludes hierarchical cache and disaggregated serving"
        )
    if (
        runner.model_config.quantization is not None
        or str(runner.model_config.dtype) != "torch.bfloat16"
    ):
        raise ValueError("Hot replacement requires unquantized BF16")
    if (
        runner.model_config.hf_config.architectures[0]
        != "Qwen4ExpForConditionalGeneration"
    ):
        raise ValueError("Initial hot replacement supports fused Qwen4Exp checkpoints")
    runner.weight_updater._assert_weight_cache_inactive("hot_swap_experts")
    metadata = get_global_expert_location_metadata()
    current = metadata.physical_to_logical_map_cpu.tolist()
    if current != expected:
        raise ValueError("The caller's expected expert mapping is stale")
    plan = plan_expert_swap(current, requested, metadata.num_logical_experts)
    modules = {
        module.layer_id: module
        for module in runner.model.modules()
        if isinstance(module, FusedMoE)
    }
    if set(modules) != set(range(metadata.num_layers)):
        raise ValueError("Every target layer must expose one routed FusedMoE")
    for module in modules.values():
        if (
            not module.runner_backend.is_triton()
            or module.num_fused_shared_experts
            or module.w13_weight.shape[0] != metadata.num_physical_experts
        ):
            raise ValueError(
                "Hot replacement requires separate shared experts and fixed compact Triton slots"
            )
        reason = getattr(module, "_derived_weight_cache_error", None)
        if reason is not None:
            raise ValueError(reason)
    needed = node_staging_bytes(plan, modules, parallel.tp_size)
    available = get_free_cpu_memory()
    if (
        max_node_staging_bytes <= 0
        or memory_reserve_bytes < 0
        or needed > max_node_staging_bytes
        or needed + memory_reserve_bytes > available
    ):
        raise MemoryError(
            f"Hot replacement needs {needed} node staging bytes plus {memory_reserve_bytes} reserve; available {available}"
        )
    old = ExpertLocationMetadata.init_by_mapping(
        runner.model_config, current, parallel.moe_ep_rank
    )
    new = ExpertLocationMetadata.init_by_mapping(
        runner.model_config, plan.physical_to_logical, parallel.moe_ep_rank
    )
    available = get_free_cpu_memory()
    if needed + memory_reserve_bytes > available:
        raise MemoryError(
            "Node staging budget no longer fits after metadata preparation"
        )
    return dict(
        runner=runner,
        metadata=metadata,
        old=old,
        new=new,
        plan=plan,
        modules=modules,
        available=available,
    )


def hot_swap_experts(
    manager,
    requested,
    expected,
    max_node_staging_bytes=64 * 2**30,
    memory_reserve_bytes=8 * 2**30,
):
    """All ranks prepare, then load, then publish; any failed mutation rolls back."""
    rows = []

    def hold_unsafe():
        manager.hot_swap_failed = True
        # Preserve the fail-safe even if pause_generation's own bookkeeping raised.
        manager.scheduler._engine_paused = True
        rows.clear()

    def vote(error, phase):
        try:
            errors = [None] * torch.distributed.get_world_size(
                group=manager.tp_cpu_group
            )
            torch.distributed.all_gather_object(
                errors, error, group=manager.tp_cpu_group
            )
        except Exception as exc:
            hold_unsafe()
            raise RuntimeError(
                f"HOT_SWAP_UNSAFE: {phase} consensus unavailable; generation held"
            ) from exc
        return "; ".join(
            f"rank{rank}: {message}" for rank, message in enumerate(errors) if message
        )

    def attempt(action, phase):
        error = ""
        try:
            action()
            torch.cuda.synchronize()
        except Exception as exc:
            error = repr(exc)
        return vote(error, phase)

    def resume():
        error = attempt(
            lambda: manager.scheduler.continue_generation(
                ContinueGenerationReqInput(torch_empty_cache=False)
            ),
            "resume",
        )
        if error:
            hold_unsafe()
            raise RuntimeError(
                f"HOT_SWAP_UNSAFE: resume failed; generation held: {error}"
            )

    context, error = None, ""
    try:
        context = _prepare_context(
            manager, requested, expected, max_node_staging_bytes, memory_reserve_bytes
        )
    except Exception as exc:
        error = repr(exc)
    error = vote(error, "preflight-before-mutation")
    if error:
        raise RuntimeError(
            f"Hot replacement preflight rejected without weight changes: {error}"
        )
    if not context["plan"].changes:
        return
    error = attempt(
        lambda: manager.scheduler.pause_generation(
            PauseGenerationReqInput(mode="in_place")
        ),
        "pause",
    )
    if error:
        hold_unsafe()
        raise RuntimeError(f"HOT_SWAP_UNSAFE: pause failed; generation held: {error}")
    rows, error = [], ""
    try:
        torch.cuda.synchronize()
        rows, _ = prepare_expert_rows(
            context["plan"],
            context["modules"],
            context["runner"].model_config.model_path,
            context["metadata"].num_logical_experts,
            get_parallel().moe_tp_rank,
            get_parallel().moe_tp_size,
            max_node_staging_bytes,
            memory_reserve_bytes,
            available_cpu_bytes=context["available"],
        )
        logger.info(
            "Hot expert replacement prepared %d changed slots, node incoming+rollback bytes=%d",
            len(rows),
            node_staging_bytes(
                context["plan"], context["modules"], get_parallel().moe_tp_size
            ),
        )
    except Exception as exc:
        error = repr(exc)
    error = vote(error, "prepare")
    if error:
        rows.clear()
        resume()
        raise RuntimeError(
            f"Hot replacement prepare failed; old weights retained: {error}"
        )
    layers = sorted({change.layer for change in context["plan"].changes})
    pointers = [
        (
            module,
            module.w13_weight.data_ptr(),
            module.w2_weight.data_ptr(),
            tuple(module.w13_weight.shape),
            tuple(module.w2_weight.shape),
        )
        for module in context["modules"].values()
    ]
    metadata_layout = _metadata_layout(context["metadata"])

    def flush():
        if not manager.flush_cache(empty_cache=False):
            raise RuntimeError("The drained cache flush was refused")

    def write():
        for row in rows:
            row.copy_new()

    def assert_stable():
        for module, w13, w2, shape13, shape2 in pointers:
            if (
                module.w13_weight.data_ptr() != w13
                or module.w2_weight.data_ptr() != w2
                or tuple(module.w13_weight.shape) != shape13
                or tuple(module.w2_weight.shape) != shape2
            ):
                raise RuntimeError("Physical expert tensor addresses or shapes changed")
        if _metadata_layout(context["metadata"]) != metadata_layout:
            raise RuntimeError(
                "Expert metadata or cached graph tensor addresses changed"
            )

    def publish():
        context["metadata"].update(context["new"], update_layer_ids=layers)
        assert_stable()

    error = attempt(flush, "flush")
    if not error:
        error = attempt(write, "write")
    if not error:
        error = attempt(publish, "publish")
    if error:

        def rollback():
            for row in rows:
                row.restore()
            context["metadata"].update(context["old"], update_layer_ids=layers)
            flush()
            assert_stable()

        rollback_error = attempt(rollback, "rollback")
        if rollback_error:
            hold_unsafe()
            raise RuntimeError(
                f"HOT_SWAP_UNSAFE: service remains paused after rollback failure: {rollback_error}; original error: {error}"
            )
        rows.clear()
        resume()
        raise RuntimeError(
            f"Hot replacement rolled back to the old expert set: {error}"
        )
    resume()
