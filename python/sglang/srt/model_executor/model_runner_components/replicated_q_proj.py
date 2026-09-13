"""Full-head Q projection weights for decode context parallelism."""

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)


@torch.no_grad()
def prepare_replicated_q_proj(*, model: nn.Module, dcp_group: "GroupCoordinator"):
    """Gather full-head weights, preserving captured storage when refreshing."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

    if dcp_group.world_size <= 1:
        return
    n_prepared = 0
    for module in model.modules():
        if not isinstance(module, DeepseekV2AttentionMLA) or module.w_kc is None:
            continue
        q_proj = module.q_b_proj if module.has_q_b_proj else module.q_proj
        if (
            module.w_kc.dtype not in (torch.bfloat16, torch.float16)
            or not isinstance(q_proj.quant_method, UnquantizedLinearMethod)
            or q_proj.weight.dtype not in (torch.bfloat16, torch.float16)
        ):
            logger.warning(
                "dcp_replicate_q_proj: skipping quantized q-proj/w_kc "
                "(bf16/fp16 only); this layer keeps the Q all-gather."
            )
            continue
        for name, shard in (
            ("w_kc_qrep", module.w_kc),
            ("q_b_proj_qrep_weight", q_proj.weight.data),
        ):
            gathered = dcp_group.all_gather(shard.contiguous(), dim=0)
            current = getattr(module, name)
            if current is None:
                setattr(module, name, gathered)
            else:
                if (
                    current.shape != gathered.shape
                    or current.dtype != gathered.dtype
                    or current.device != gathered.device
                ):
                    raise RuntimeError(
                        f"Replicated Q projection layout changed: {name}"
                    )
                current.copy_(gathered)
        n_prepared += 1
    logger.info(
        "dcp_replicate_q_proj: prepared full-head Q weights for %d MLA layers",
        n_prepared,
    )
