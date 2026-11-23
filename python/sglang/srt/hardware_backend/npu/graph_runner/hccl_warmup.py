"""Initialize the lazy communication domains before CANN SMLA metadata warmup."""

import torch


def prepare_hccl_before_metadata(model, parallel):
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher

    dispatchers = []
    for module in model.modules():
        wrapper = getattr(module, "dispatcher", None)
        for impl in getattr(wrapper, "_inners", (wrapper,)):
            if isinstance(impl, DeepEPDispatcher):
                dispatchers.append(impl)
    if not dispatchers:
        return

    # Some coordinator roles alias the same ProcessGroup. All ranks visit the
    # roles in the same order; process-local identity is only used to deduplicate.
    ready = torch.zeros(1, dtype=torch.int32, device="npu")
    seen = set()
    for role in ("tp_group", "attn_tp_group", "moe_ep_group"):
        group = getattr(getattr(parallel, role, None), "device_group", None)
        if group is not None and group not in seen:
            seen.add(group)
            if torch.distributed.get_world_size(group) > 1:
                torch.distributed.all_reduce(ready, group=group)
    torch.npu.synchronize()
    largest = max(dispatchers, key=lambda d: d._low_latency_dispatcher.num_experts)
    if largest.prebuild_pending:
        largest.prebuild()
    torch.npu.synchronize()
