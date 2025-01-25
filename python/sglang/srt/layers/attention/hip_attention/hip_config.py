import warnings
from dataclasses import InitVar, dataclass, field
from typing import List, Optional, Union

from hip.models.hip_attention.gen3.attention_metadata import ScanStage

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
        stage_chunk_size=8,
        stage_k=8192,
        stage_stride=1,
    ),
]


@dataclass
class HiPAttentionPerLayerConfig:
    second_stage_k: int = 2048
    sliding_window_size: int = 1024
    sink_token_size: int = 256
    sa_extend_backend: str = "streaming"
    scan_extend_backend: Optional[str] = None
    stages: list[ScanStage] = field(default_factory=lambda: _DEFAULT_STAGES)

    parsed_json: InitVar[dict | None] = None

    def __post_init__(self, parsed_json: dict | None):
        super().__init__()
        if parsed_json is not None:
            if "second_stage_k" in parsed_json:
                self.second_stage_k = parsed_json["second_stage_k"]
                parsed_json.pop("second_stage_k")
            if "sliding_window_size" in parsed_json:
                self.sliding_window_size = parsed_json["sliding_window_size"]
                parsed_json.pop("sliding_window_size")
            if "sink_token_size" in parsed_json:
                self.sink_token_size = parsed_json["sink_token_size"]
                parsed_json.pop("sink_token_size")
            if "sa_extend_backend" in parsed_json:
                self.sa_extend_backend = parsed_json["sa_extend_backend"]
                parsed_json.pop("sa_extend_backend")
            if "scan_extend_backend" in parsed_json:
                self.scan_extend_backend = parsed_json["scan_extend_backend"]
                parsed_json.pop("scan_extend_backend")
            if "stages" in parsed_json:
                self.stages = [ScanStage(**stage) for stage in parsed_json["stages"]]
                parsed_json.pop("stages")
            if parsed_json:
                raise ValueError(f"Unknown keys in json: {parsed_json.keys()}")


@dataclass
class HiPAttentionConfig:
    dense_layers: list[int] = field(default_factory=lambda: [0, 1, 2])
    block_sparse_block_size_q: int = 64
    metadata_cache_max_batch_size: int = 32
    mask_refresh_interval: Union[int, List[int]] = field(
        default_factory=lambda: [32, 16, 8]
    )
    using_extend: bool = True
    layers: list[HiPAttentionPerLayerConfig] = field(
        default_factory=lambda: [
            HiPAttentionPerLayerConfig(
                parsed_json={
                    "second_stage_k": 4096,
                    "sliding_window_size": 1024,
                    "sink_token_size": 256,
                }
            ),
            HiPAttentionPerLayerConfig(),
        ]
    )
    prefill_layers: list[HiPAttentionPerLayerConfig] = field(
        default_factory=lambda: [
            HiPAttentionPerLayerConfig(
                parsed_json={
                    "second_stage_k": 4096,
                    "sliding_window_size": 1024,
                    "sink_token_size": 256,
                }
            ),
            HiPAttentionPerLayerConfig(),
        ]
    )

    # deprecated
    apply_v_dot: bool = False
    prefill_always_dense: bool = False
    decode_always_dense: bool = False
    force_dense: bool = False
    prefill_dense_threshold: int = 8192

    parsed_json: InitVar[dict | None] = None

    def __post_init__(self, parsed_json: dict | None):
        super().__init__()

        if parsed_json is not None:
            if "apply_v_dot" in parsed_json:
                self.apply_v_dot = parsed_json["apply_v_dot"]
                parsed_json.pop("apply_v_dot")
            if "dense_layers" in parsed_json:
                self.dense_layers = parsed_json["dense_layers"]
                parsed_json.pop("dense_layers")
            if "prefill_always_dense" in parsed_json:
                self.prefill_always_dense = parsed_json["prefill_always_dense"]
                parsed_json.pop("prefill_always_dense")
            if "decode_always_dense" in parsed_json:
                self.decode_always_dense = parsed_json["decode_always_dense"]
                parsed_json.pop("decode_always_dense")
            if "force_dense" in parsed_json:
                self.force_dense = parsed_json["force_dense"]
                parsed_json.pop("force_dense")
            if "prefill_dense_threshold" in parsed_json:
                self.prefill_dense_threshold = parsed_json["prefill_dense_threshold"]
                parsed_json.pop("prefill_dense_threshold")
            if "block_sparse_block_size_q" in parsed_json:
                self.block_sparse_block_size_q = parsed_json[
                    "block_sparse_block_size_q"
                ]
                parsed_json.pop("block_sparse_block_size_q")
            if "metadata_cache_max_batch_size" in parsed_json:
                self.metadata_cache_max_batch_size = parsed_json[
                    "metadata_cache_max_batch_size"
                ]
                parsed_json.pop("metadata_cache_max_batch_size")
            if "mask_refresh_interval" in parsed_json:
                assert isinstance(parsed_json["mask_refresh_interval"], (int, list))
                self.mask_refresh_interval = parsed_json["mask_refresh_interval"]
                parsed_json.pop("mask_refresh_interval")
            if "using_extend" in parsed_json:
                self.using_extend = parsed_json["using_extend"]
                parsed_json.pop("using_extend")
            if "layers" in parsed_json:
                self.layers = [
                    HiPAttentionPerLayerConfig(parsed_json=layer)
                    for layer in parsed_json["layers"]
                ]
                self.prefill_layers = self.layers
                parsed_json.pop("layers")
            if "prefill_layers" in parsed_json:
                self.prefill_layers = [
                    HiPAttentionPerLayerConfig(parsed_json=layer)
                    for layer in parsed_json["prefill_layers"]
                ]
                parsed_json.pop("prefill_layers")
            if parsed_json:
                raise ValueError(f"Unknown keys in json: {parsed_json.keys()}")

        num_stages = len(self.layers[0].stages)
        for layer_config in self.layers:
            assert num_stages == len(layer_config.stages)

        if isinstance(self.mask_refresh_interval, int):
            self.mask_refresh_interval = [
                self.mask_refresh_interval,
            ] * num_stages
