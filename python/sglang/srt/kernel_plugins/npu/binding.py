"""Install the registered role after weights and KV are ready, before runner warmup."""

import logging

import torch.distributed as dist
from sglang.kernels.plugin_kernels.npu.spec import PreparationView, find_spec
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def install_megakernels(runner):
    cfg = resolved_view(runner.server_args)
    architecture = type(runner.model).__name__
    group = get_parallel().tp_group.cpu_group
    binding = error = None
    try:
        spec = find_spec(architecture)
        if spec is not None and (spec.role == "nextn") == runner.is_draft_worker:
            resources = PreparationView(
                model=runner.model,
                token_to_kv_pool=runner.token_to_kv_pool,
                physical_slots=not runner.runner_ids.is_translating
                and not runner.token_to_kv_pool_allocator.hands_out_virtual_ids,
            )
            binding = spec.prepare(resources, cfg)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    signature = (
        architecture,
        getattr(binding, "role", None),
        getattr(binding, "mode", None),
        cfg.speculative_num_steps,
        cfg.speculative_num_draft_tokens,
        error,
    )
    ranks = [None] * dist.get_world_size(group)
    dist.all_gather_object(ranks, signature, group=group)
    if any(peer[-1] for peer in ranks):
        raise RuntimeError(f"NPU megakernel preflight failed: {ranks}")
    if any(peer[1] is None for peer in ranks):
        logger.info("NPU megakernel declined %s on one or more TP ranks", architecture)
        return None
    if any(peer != signature for peer in ranks):
        raise RuntimeError(f"NPU megakernel preflight failed: {ranks}")
    if binding is not None:
        binding.build(runner.model)
    logger.info("NPU megakernel architecture=%s role=%s mode=%s", *signature[:3])
    return binding
