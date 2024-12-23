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
class HipAttentionPerLayerConfig:
    second_stage_k: int = 2048
    sliding_window_size: int = 1024
    sink_token_size: int = 256
    sa_extend_backend: str = 'streaming'
    stages: list = field(default_factory=lambda: _DEFAULT_STAGES)

    def __init__(self, parsed_json):
        self.second_stage_k = parsed_json['second_stage_k']
        self.sliding_window_size = parsed_json['sliding_window_size']
        self.sink_token_size = parsed_json['sink_token_size']
        self.sa_extend_backend = parsed_json['sa_extend_backend']
        self.stages = [
            ScanStage(**stage)
            for stage in parsed_json['stages']
        ]


@dataclass
class HipAttentionConfig:
    extend_context_length: int = 32768
    apply_v_dot: bool = False
    dense_layers: list[int] = field(default_factory=lambda: [0, 1, 2])
    prefill_always_dense: bool = False
    force_dense: bool = False
    prefill_dense_threshold: int = 8192
    decode_dense_threshold: int = 8192
    layers: list[HipAttentionPerLayerConfig] = None

    def __init__(self, parsed_json):
        self.extend_context_length = parsed_json['extend_context_length']
        self.apply_v_dot = parsed_json['apply_v_dot']
        self.layers = [
            HipAttentionPerLayerConfig(layer)
            for layer in parsed_json['layers']
        ]
