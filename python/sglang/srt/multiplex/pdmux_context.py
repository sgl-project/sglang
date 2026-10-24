from dataclasses import dataclass, field
from typing import List

import torch
import yaml

STREAM_GROUPS = []
SM_COUNTS = []
SM_GROUP_NUM = 8  # Default number of SM groups
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None
_RESERVED_GREEN_STREAMS = []
_FULL_DEVICE_DECODE_STREAMS = set()

_OVERLAP_DECODE_STREAM_PRIORITY = -1


@dataclass
class PDMuxConfig:
    sm_group_num: int = 8
    manual_divisions: List[List[int]] = field(
        default_factory=list
    )  # [prefill_sm, decode_sm, decode_bs_threshold]
    split_forward_token_budget: int = 65536
    # Zero keeps the token-budget-only policy. A positive cap bounds CPU
    # submission before the next decode step, even for short prefills.
    max_split_forward_layers: int = 0
    decode_bs_divisor: int = 36
    # Overlap mode: prefill keeps its green-context SM cap while decode runs on
    # plain full-device streams, so the two SM sets deliberately overlap and the
    # manual_divisions decode_sm column is ignored.
    overlap_decode_full_sm: bool = False


def load_pdmux_config(config_path: str) -> PDMuxConfig:
    """Load pdmux configuration from YAML file into a dataclass."""
    if not config_path:
        return PDMuxConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if "sm_group_num" not in raw:
        raise ValueError("Missing required field: sm_group_num")

    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be >= 3")

    manual_divisions = raw.get("manual_divisions", [])

    expected = raw["sm_group_num"] - 2
    if manual_divisions and len(manual_divisions) != expected:
        raise ValueError(
            f"manual_divisions must have {expected} entries, "
            f"but got {len(manual_divisions)}"
        )

    overlap_decode_full_sm = raw.get("overlap_decode_full_sm", False)
    if overlap_decode_full_sm and not manual_divisions:
        raise ValueError(
            "overlap_decode_full_sm requires manual_divisions: the "
            "automatic divide_sm split enforces prefill >= 50% for mutually "
            "exclusive partitions, which does not describe an overlapped one."
        )

    previous_threshold = None
    for i, division in enumerate(manual_divisions):
        if len(division) != 3:
            raise ValueError(
                "manual_divisions entries must be "
                "[prefill_sm, decode_sm, decode_bs_threshold]"
            )
        prefill_sm, decode_sm, threshold = division
        if prefill_sm <= 0:
            raise ValueError(f"manual_divisions[{i}] prefill_sm must be positive")
        if overlap_decode_full_sm:
            if decode_sm < 0:
                raise ValueError(
                    f"manual_divisions[{i}] decode_sm must be non-negative "
                    "when overlap_decode_full_sm is enabled"
                )
        elif decode_sm <= 0:
            raise ValueError(f"manual_divisions[{i}] decode_sm must be positive")
        if threshold < 0:
            raise ValueError(
                f"manual_divisions[{i}] decode_bs_threshold must be non-negative"
            )
        if previous_threshold is not None and threshold <= previous_threshold:
            raise ValueError(
                "manual_divisions decode_bs_threshold values must be "
                "strictly increasing"
            )
        previous_threshold = threshold

    split_forward_token_budget = raw.get("split_forward_token_budget", 65536)
    max_split_forward_layers = raw.get("max_split_forward_layers", 0)
    decode_bs_divisor = raw.get("decode_bs_divisor", 36)
    if split_forward_token_budget <= 0:
        raise ValueError("split_forward_token_budget must be positive")
    if decode_bs_divisor <= 0:
        raise ValueError("decode_bs_divisor must be positive")
    if (
        not isinstance(max_split_forward_layers, int)
        or isinstance(max_split_forward_layers, bool)
        or max_split_forward_layers < 0
    ):
        raise ValueError("max_split_forward_layers must be a non-negative integer")

    return PDMuxConfig(
        sm_group_num=raw["sm_group_num"],
        manual_divisions=manual_divisions,
        split_forward_token_budget=split_forward_token_budget,
        max_split_forward_layers=max_split_forward_layers,
        decode_bs_divisor=decode_bs_divisor,
        overlap_decode_full_sm=overlap_decode_full_sm,
    )


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
    possible_values = [
        x
        for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
        if x >= total_sms - x and total_sms - x >= 16
    ]
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


def initialize_stream_groups(gpu_id: int, config: PDMuxConfig):
    from sgl_kernel import spatial

    global \
        STREAM_GROUPS, \
        SM_COUNTS, \
        SM_GROUP_NUM, \
        CURRENT_STREAM_IDX, \
        CURRENT_STREAM_GROUP
    global _RESERVED_GREEN_STREAMS, _FULL_DEVICE_DECODE_STREAMS
    # for pd_multiplexing, Init stream_groups
    device = torch.cuda.current_device()
    total_sm_count = spatial.get_sm_available(gpu_id)
    # (prefill_sm_count, decode_sm_count)
    if config.manual_divisions:
        divisions = [
            (prefill_sm, decode_sm)
            for prefill_sm, decode_sm, _ in config.manual_divisions
        ]
    else:
        divisions = divide_sm(
            total_sm_count,
            torch.cuda.get_device_capability(device),
            config.sm_group_num - 2,
        )

    if config.overlap_decode_full_sm:
        for prefill_sm, _ in divisions:
            if not 0 < prefill_sm < total_sm_count:
                raise ValueError(
                    f"overlap_decode_full_sm needs a prefill_sm strictly inside "
                    f"(0, {total_sm_count}); got {prefill_sm}. A full-device cap "
                    f"leaves decode no SMs of its own."
                )
        # Decode reaches every SM, so its column records the full device.
        divisions = [(prefill_sm, total_sm_count) for prefill_sm, _ in divisions]
    else:
        for prefill_sm, decode_sm in divisions:
            if prefill_sm + decode_sm != total_sm_count:
                raise ValueError(
                    "exclusive PDMux manual divisions must assign every SM: "
                    f"prefill_sm ({prefill_sm}) + decode_sm ({decode_sm}) "
                    f"must equal the device SM count ({total_sm_count}). "
                    "Use overlap_decode_full_sm for full-device decode."
                )

    SM_COUNTS = []
    SM_COUNTS.append((total_sm_count, 0))  # Normal stream for prefill
    SM_COUNTS.extend(divisions)  # Add the divided SM counts
    SM_COUNTS.append((0, total_sm_count))  # Normal stream for decode
    STREAM_GROUPS = []
    _RESERVED_GREEN_STREAMS = []
    _FULL_DEVICE_DECODE_STREAMS = set()
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for prefill
    for prefill_sm, decode_sm in divisions:
        if config.overlap_decode_full_sm:
            prefill_stream, reserved_stream = spatial.create_greenctx_stream_by_value(
                prefill_sm, total_sm_count - prefill_sm, gpu_id
            )
            _RESERVED_GREEN_STREAMS.append(reserved_stream)
            decode_stream = torch.cuda.Stream(
                gpu_id, priority=_OVERLAP_DECODE_STREAM_PRIORITY
            )
            STREAM_GROUPS.append((prefill_stream, decode_stream))
            _FULL_DEVICE_DECODE_STREAMS.add(decode_stream.cuda_stream)
        else:
            STREAM_GROUPS.append(
                (spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id))
            )
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for decode
    _FULL_DEVICE_DECODE_STREAMS.add(STREAM_GROUPS[-1][1].cuda_stream)

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


def get_pdmux_decode_alt_stream(alt_stream):
    """Allow ordinary helper streams only on a full-device decode lane.

    Inspect the actual stream: graph capture iterates stream groups without
    changing CURRENT_STREAM_IDX. A helper on a green decode lane would escape
    the SM partition just as it would on a green prefill lane.
    """
    if alt_stream is None:
        return None
    if torch.cuda.current_stream().cuda_stream in _FULL_DEVICE_DECODE_STREAMS:
        return alt_stream
    return None
