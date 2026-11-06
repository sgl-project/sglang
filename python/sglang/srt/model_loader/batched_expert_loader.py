"""Bounded host staging for the initial load of serialized Kimi MXFP4 experts."""

import logging
from collections import deque
from contextlib import contextmanager

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def can_batch_expert_load(model):
    return (
        envs.SGLANG_ENABLE_BATCHED_EXPERT_LOAD.get()
        and type(model).__name__ == "KimiK31ForConditionalGeneration"
    )


class _StagingPool:
    def __init__(self, budget):
        self.budget = budget
        self.used = 0
        self.pending = deque()
        self.copies = 0
        self.bytes = 0

    def reserve(self, size):
        while self.pending and (
            self.pending[0][0].query() or self.used + size > self.budget
        ):
            event, host, nbytes = self.pending.popleft()
            event.synchronize()
            self.used -= nbytes
        if self.used + size > self.budget:
            return False
        self.used += size
        return True

    def submit(self, param, host):
        param.copy_(host, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(param.device))
        nbytes = host.numel() * host.element_size()
        self.pending.append((event, host, nbytes))
        self.copies += 1
        self.bytes += nbytes

    def finish(self):
        while self.pending:
            event, host, size = self.pending.popleft()
            event.synchronize()
            self.used -= size


class _ExpertParameterLoader:
    def __init__(self, layer, name, param, pool):
        self.layer = layer
        self.name = name
        self.param = param
        self.original = param.weight_loader
        self.pool = pool
        self.shadow = None
        self.seen = set()
        self.shards = {"w1", "w3"} if name.startswith("w13") else {"w2"}
        self.expected = param.shape[0] * len(self.shards)
        self.bypass = False

    def __call__(self, param, loaded_weight, weight_name, shard_id, expert_id):
        if self.bypass or expert_id is None:
            if self.shadow is not None:
                raise ValueError("Cannot mix fused and individual expert checkpoints")
            self.bypass = True
            return self.original(param, loaded_weight, weight_name, shard_id, expert_id)
        local_id = self.layer._map_global_expert_id_to_local_expert_id(expert_id)
        if local_id == -1:
            return
        if shard_id not in self.shards or loaded_weight.device.type != "cpu":
            raise ValueError("Unsupported tensor in batched serialized expert load")
        if self.shadow is None:
            size = param.numel() * param.element_size()
            if not self.pool.reserve(size):
                # Stay on the direct path for this parameter: a later staging
                # allocation must not overwrite already-loaded GPU slices.
                self.bypass = True
                return self.original(
                    param, loaded_weight, weight_name, shard_id, expert_id
                )
            host = torch.empty_like(param, device="cpu", pin_memory=True)
            # Only unpadded parameters are admitted below. Every element is
            # overwritten before submission, so no whole-buffer fill is needed.
            self.shadow = torch.nn.Parameter(host, requires_grad=False)
            self.shadow.__dict__.update(param.__dict__)
        self.original(self.shadow, loaded_weight, weight_name, shard_id, expert_id)
        self.seen.add((local_id, shard_id))
        if len(self.seen) == self.expected:
            self.pool.submit(param, self.shadow)
            self.shadow = None
            self.seen.clear()
            # Subsequent duplicate tensors retain the normal last-write behavior.
            self.bypass = True


def _restore_parameter_loaders(loaders):
    for loader in loaders:
        loader.param.weight_loader = loader.original
        loader.shadow = None
    # Postprocessing replaces parameters with packed/shuffled tensors. Keeping
    # these wrappers alive would retain all of the original GPU allocations.
    loaders.clear()


@contextmanager
def batch_expert_weight_loading(model, budget=4 * 1024**3, num_threads=8):
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    pool = _StagingPool(budget)
    loaders = []
    original_num_threads = torch.get_num_threads()
    try:
        for layer in model.modules():
            if not isinstance(layer, FusedMoE):
                continue
            if (
                layer.quant_config is None
                or layer.quant_config.get_name() != "mxfp4"
                or not layer.quant_config.is_static_cfg()
                or layer._has_fused_shared
                or layer.w13_weight.shape[1]
                != 2 * layer.intermediate_size_per_partition
                or layer.w2_weight.shape[2]
                != layer.intermediate_size_per_partition // 2
                or (
                    layer.uses_expert_location
                    and get_global_expert_location_metadata() is not None
                )
            ):
                continue
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                param = getattr(layer, name, None)
                if (
                    param is None
                    or param.device.type != "cuda"
                    or param.dtype != torch.uint8
                    or param.ndim != 3
                    or getattr(param, "_sglang_require_global_experts", False)
                ):
                    continue
                loader = _ExpertParameterLoader(layer, name, param, pool)
                loaders.append(loader)
                param.weight_loader = loader
        # Do not retain the final parameter through these generator locals.
        param = loader = None
        if loaders:
            torch.set_num_threads(num_threads)

        def check_complete():
            incomplete = [
                loader.name for loader in loaders if loader.shadow is not None
            ]
            if incomplete:
                raise ValueError(
                    f"Incomplete expert tensors in checkpoint: {incomplete[:8]}"
                )

        def complete_weights(weights):
            yield from weights
            check_complete()
            pool.finish()
            _restore_parameter_loaders(loaders)
            torch.set_num_threads(original_num_threads)

        yield complete_weights
        check_complete()
    finally:
        try:
            pool.finish()
        finally:
            _restore_parameter_loaders(loaders)
            torch.set_num_threads(original_num_threads)
        logger.info(
            "Batched expert load: %d H2D copies, %.3f GiB",
            pool.copies,
            pool.bytes / 1024**3,
        )
