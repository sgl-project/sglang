import torch

from sglang.srt.distributed import GroupCoordinator, get_tp_group

_ATTN_TP_GROUP = None
_ATTN_TP_RANK = None
_ATTN_TP_SIZE = None
_DP_RANK = None
_DP_SIZE = None


def compute_dp_attention_world_info(enable_dp_attention, tp_rank, tp_size, dp_size):
    if not enable_dp_attention:
        return tp_rank, tp_size, 0

    attn_tp_size = tp_size // dp_size
    dp_rank = tp_rank // attn_tp_size
    attn_tp_rank = tp_rank % attn_tp_size
    return attn_tp_rank, attn_tp_size, dp_rank


def initialize_dp_attention(enable_dp_attention, tp_rank, tp_size, dp_size):
    global _ATTN_TP_GROUP, _ATTN_TP_RANK, _ATTN_TP_SIZE, _DP_RANK, _DP_SIZE

    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

    _ATTN_TP_RANK, _ATTN_TP_SIZE, _DP_RANK = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size
    )
    _DP_SIZE = dp_size

    tp_group = get_tp_group()
    _ATTN_TP_GROUP = GroupCoordinator(
        [
            list(range(head, head + _ATTN_TP_SIZE))
            for head in range(0, tp_size, _ATTN_TP_SIZE)
        ],
        tp_rank,
        torch.distributed.get_backend(tp_group.device_group),
        SYNC_TOKEN_IDS_ACROSS_TP,
        False,
        False,
        False,
        False,
        group_name="attention_tp",
    )


def get_attention_tp_group():
    assert _ATTN_TP_GROUP is not None, "dp attention not initialized!"
    return _ATTN_TP_GROUP


def get_attention_tp_rank():
    assert _ATTN_TP_RANK is not None, "dp attention not initialized!"
    return _ATTN_TP_RANK


def get_attention_tp_size():
    assert _ATTN_TP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_TP_SIZE


def get_attention_dp_rank():
    assert _DP_RANK is not None, "dp attention not initialized!"
    return _DP_RANK


def get_attention_dp_size():
    assert _DP_SIZE is not None, "dp attention not initialized!"
    return _DP_SIZE
