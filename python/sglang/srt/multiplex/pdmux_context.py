import torch
from sgl_kernel import spatial

STREAM_GROUPS = []
SM_COUNTS = []
SM_GROUP_NUM = 8  # Default number of SM groups
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None


def get_arch_constraints(compute_capability):
    major, minor = compute_capability
    # green context constraints for different architectures
    if major == 6:
        return 1, 1  # min_per_part, multiple
    elif major == 7:
        return 2, 2
    elif major == 8:
        return 4, 2
    elif major == 9 and minor >= 0:
        return 8, 8
    else:
        raise ValueError(f"Unsupported compute capability: {major}.{minor}")

def divide_sm(total_sms, compute_capability, groups):
    """
    :param total_sms: total sm count on a single GPU
    :param compute_capability: (major, minor)
    :return: SM partition group(prefill sm, decode sm)
    """
    min_per_part, multiple = get_arch_constraints(compute_capability)    
    
    possible_values = [x for x in range(min_per_part, total_sms - min_per_part + 1, multiple) if x >= total_sms - x and total_sms - x >= 16]
    
    if not possible_values:
        raise ValueError(
            f"No valid partitions found for total SMs {total_sms} "
            f"with constraints (min per part: {min_per_part}, multiple: {multiple})"
        )
    
    if len(possible_values) >= groups:
        step = max(1, len(possible_values) // groups)
        selected_values = possible_values[::step][:groups]
    else:
        selected_values = possible_values
    
    divisions = []
    for part1 in selected_values:
        part2 = total_sms - part1
        divisions.append((part1, part2))
        
    divisions.reverse()  # Reverse to have larger prefill SM first
    
    return divisions

def initialize_stream_groups(gpu_id: int, groups: int):
    global STREAM_GROUPS, SM_COUNTS, SM_GROUP_NUM, CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    # for pd_multiplexing, Init stream_groups
    device = torch.cuda.current_device()
    total_sm_count = spatial.get_sm_available(gpu_id)
    # (prefill_sm_count, decode_sm_count)
    divsions = divide_sm(total_sm_count, torch.cuda.get_device_capability(device), groups)
    SM_COUNTS = []
    SM_COUNTS.append((total_sm_count, 0))  # Normal stream for prefill
    SM_COUNTS.extend(divsions)  # Add the divided SM counts
    SM_COUNTS.append((0, total_sm_count))  # Normal stream for decode
    STREAM_GROUPS = []
    STREAM_GROUPS.append((torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id)))  # Normal stream for prefill
    for prefill_sm, decode_sm in divsions:
        STREAM_GROUPS.append((spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id)))
    STREAM_GROUPS.append((torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id)))  # Normal stream for decode

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
