"""Release unreachable modules after GLM AFD role projection, before loading."""

import gc
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)


def collect_projection_garbage(model, role):
    # Native and local NextN execution never need this preparation operation.
    if role not in ("attention", "ffn"):
        return None
    device = next(model.parameters()).device
    cuda = device.type == "cuda"

    def snapshot():
        if not cuda:
            return {"allocated_bytes": None, "reserved_bytes": None, "free_bytes": None}
        return {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "free_bytes": torch.cuda.mem_get_info(device)[0],
        }

    if cuda:
        torch.cuda.synchronize(device)
    before = snapshot()
    # Parameter.weight_loader may be a bound method of a discarded FFN.
    # Collect only unreachable cycles; reachable routers/weights stay intact.
    collected = gc.collect()
    if cuda:
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    after = snapshot()
    receipt = {
        "phase": "after_role_projection_before_weight_loading",
        "role": role,
        "rank": (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else None
        ),
        "pid": os.getpid(),
        "device": str(device),
        "gc_collected_objects": collected,
        "before": before,
        "after": after,
    }
    logger.info("AFD_GLM_PROJECTION_GC %s", json.dumps(receipt, sort_keys=True))
    return receipt
