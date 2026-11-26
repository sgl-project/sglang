from sglang.srt.layers.linear import MergedColumnParallelLinear


class Inkling2QKVRLinear(MergedColumnParallelLinear):
    SHARD_K = 0
    SHARD_V = 1
    SHARD_Q = 2
    SHARD_R = 3
    SHARD_GATE = 4
    SHARD_SINK = 5
    SHARD_PAD = 6
    PROJECTION_NAMES = (
        "wk_dv",
        "wv_dv",
        "wq_du",
        "wr_du",
        "gate_attn",
        "attn_sink_proj",
    )
