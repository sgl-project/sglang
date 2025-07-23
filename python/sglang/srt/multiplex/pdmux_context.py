import torch
from sgl_kernel import spatial

STREAM_GROUPS = []
SM_COUNTS = []
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None


def initialize_stream_groups(gpu_id: int):
    global STREAM_GROUPS, SM_COUNTS, CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    # for pd_multiplexing, Init stream_groups
    total_sm_count = spatial.get_sm_available(gpu_id)
    SM_COUNTS = [
        # (prefill_sm_count, decode_sm_count)
        (total_sm_count, 0),
        (total_sm_count // 2, total_sm_count - total_sm_count // 2),
        (0, total_sm_count),
    ]
    STREAM_GROUPS = [
        # (prefill_stream, decode_stream)
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id)),
        spatial.create_greenctx_stream_by_value(
            total_sm_count // 2, total_sm_count - total_sm_count // 2, gpu_id
        ),
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id)),
    ]

    CURRENT_STREAM_IDX = 0
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def set_current_stream_idx(idx: int):
    global CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    if idx < 0 or idx >= len(STREAM_GROUPS):
        raise ValueError(f"Invalid stream index: {idx}")
    CURRENT_STREAM_IDX = idx
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def get_stream_groups() -> list[tuple[torch.cuda.Stream, torch.cuda.Stream]]:
    """Get the stream groups."""
    return STREAM_GROUPS


def get_sm_counts() -> list[tuple[int, int]]:
    """Get the SM counts."""
    return SM_COUNTS


def get_current_stream_idx() -> int:
    """Get the current stream index."""
    return CURRENT_STREAM_IDX
