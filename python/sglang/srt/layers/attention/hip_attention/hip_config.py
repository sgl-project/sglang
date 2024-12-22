from dataclasses import dataclass, field

from hip.models.hip_attention.attention2_draft_sampling_extend import ScanStage


_DEFAULT_STAGES = [
    ScanStage(
        stage_block_size_q=64,
        stage_block_stride_q=4,
        stage_chunk_size=256,
        stage_k=None,
        stage_stride=1,
    ),
    ScanStage(
        stage_block_size_q=64,
        stage_block_stride_q=4,
        stage_chunk_size=32,
        stage_k=32768,
        stage_stride=1,
    ),
    ScanStage(
        stage_block_size_q=64,
        stage_block_stride_q=1,
        stage_chunk_size=16,
        stage_k=8192,
        stage_stride=1,
    ),
]


@dataclass
class HipAttentionConfig:
    dense_layers: list[int] = field(default_factory=lambda: [0, 1, 2])
    prefill_always_dense: bool = False
    force_dense: bool = False
    prefill_dense_threshold: int = 8192
    second_k: int = 2048
    second_k_dense: int = 4096
    extend_context_length: int = 32768
    sa_extend_backend: str = 'streaming'
    stages: list = field(default_factory=lambda: _DEFAULT_STAGES)
    apply_v_dot: bool = False

    def __init__(self, parsed_json):
        self.dense_layers = parsed_json["dense_layers"]
        # TODO: Add more fields here
