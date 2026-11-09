import logging

import torch

logger = logging.getLogger(__name__)


def cap_torch_allocator_for_headroom(*, device: int, headroom_gb: float) -> None:
    """Keep ``headroom_gb`` free for non-torch allocations: at the cap torch releases its cached
    (often fragmented) blocks and retries, instead of starving e.g. HIP scratch growth."""
    if headroom_gb <= 0:
        return
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    non_torch = total - free - reserved
    cap = total - non_torch - int(headroom_gb * (1 << 30))
    if cap <= reserved:
        logger.warning(
            "Torch allocator headroom of %.2f GB does not fit: %.2f GB free on the device; "
            "leaving the allocator uncapped",
            headroom_gb,
            free / (1 << 30),
        )
        return
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    logger.info(
        "Torch allocator capped at %.2f GB (%.2f GB reserved now, %.2f GB outside torch), "
        "keeping %.2f GB free for non-torch allocations",
        cap / (1 << 30),
        reserved / (1 << 30),
        non_torch / (1 << 30),
        headroom_gb,
    )
