"""Install the registered role after weights and KV are ready, before runner warmup."""

import logging

import torch.distributed as dist

from sglang.kernels.plugin_kernels.npu.spec import Decline, PreparationView, find_spec
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def _decline_text(reason):
    return reason.reason if isinstance(reason, Decline) else reason


def _unexpected_decline(reason):
    if isinstance(reason, Decline):
        return not reason.expected
    return True


def install_megakernels(runner):
    architecture = type(runner.model).__name__
    group = get_parallel().tp_group.cpu_group
    binding = error = decline = None
    cfg = None
    try:
        cfg = resolved_view(runner.server_args)
        spec = find_spec(architecture)
        if spec is None:
            decline = f"no megakernel spec for {architecture}"
        elif (spec.role == "nextn") != runner.is_draft_worker:
            decline = (
                f"spec role {spec.role} does not match "
                f"draft_worker={runner.is_draft_worker}"
            )
        else:
            resources = PreparationView(
                model=runner.model,
                token_to_kv_pool=runner.token_to_kv_pool,
                physical_slots=not runner.runner_ids.is_translating
                and not runner.token_to_kv_pool_allocator.hands_out_virtual_ids,
            )
            outcome = spec.prepare(resources, cfg)
            if isinstance(outcome, Decline):
                decline = outcome
            elif outcome is None:
                decline = None
            else:
                binding = outcome
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    signature = (
        architecture,
        getattr(binding, "role", None),
        getattr(binding, "mode", None),
        getattr(cfg, "speculative_num_steps", None),
        getattr(cfg, "speculative_num_draft_tokens", None),
        error,
        decline,
    )
    ranks = [None] * dist.get_world_size(group)
    dist.all_gather_object(ranks, signature, group=group)
    if any(peer[5] for peer in ranks):
        raise RuntimeError(f"NPU megakernel preflight failed: {ranks}")
    if any(peer[1] is None for peer in ranks):
        reasons = [peer[6] for peer in ranks]
        # Decline.expected is an explicit off switch. Any other reason means
        # the enabled plugin did not install.
        unexpected = any(_unexpected_decline(reason) for reason in reasons)
        log = logger.warning if unexpected else logger.info
        log(
            "NPU megakernel declined %s: %s",
            architecture,
            [_decline_text(reason) for reason in reasons],
        )
        return None
    if any(peer != signature for peer in ranks):
        raise RuntimeError(f"NPU megakernel preflight failed: {ranks}")
    if binding is not None:
        binding.build(runner.model)
    logger.info("NPU megakernel architecture=%s role=%s mode=%s", *signature[:3])
    return binding
