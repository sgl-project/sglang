from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from sglang_simulator.spec import AcceleratorInfo, DataType


@dataclass
class SchedulerConfig:
    data_type: Optional[DataType] = (
        None  # Data type for model weights and activations. If none is set, it will be automatically detected.
    )
    kv_cache_data_type: Optional[DataType] = None
    # AIC adapter overrides — bypass MAP_DTYPE_TO_* lookup when set.
    # Pass aiconfigurator MoEQuantMode/FMHAQuantMode/CommQuantMode enum name as string
    # (e.g. 'w4a8_mxfp4_mxfp8' for DSv4-Pro on Blackwell).
    moe_quant_mode_override: Optional[str] = None
    fmha_quant_mode_override: Optional[str] = None
    comm_quant_mode_override: Optional[str] = None
    mem_fraction_static: Optional[float] = None
    max_total_tokens: Optional[int] = None

    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    cp_style: str = "none"

    # DSv4 KV cache calculator inputs (sourced from server_args)
    page_size: Optional[int] = None
    swa_full_tokens_ratio: Optional[float] = None

    # Optional explicit override of per-GPU KV bytes/token, sourced from
    # sglang server startup log: "KV Cache is allocated. #tokens: N, KV size: G GB"
    #   kv_bytes_per_token_per_gpu = G * 1024**3 / N
    # When set, takes priority over the model-derived calculator path.
    # Useful for models where sglang doesn't expose its KV calculator output
    # (e.g. GlmMoeDsa) and we want metrics to match the live sglang server.
    kv_bytes_per_token_per_gpu: Optional[float] = None

    # L2 host KV pool sizing: host_pool_tokens = hicache_ratio * max_total_tokens.
    hicache_ratio: Optional[float] = None
    enable_hierarchical_cache: Optional[bool] = None

    # framework backend
    backend_name: str = "sglang"
    backend_version: Optional[str] = None

    @property
    def attn_tp_size(self) -> int:
        divisor = self.dp_size * self.cp_size
        if self.tp_size % divisor != 0:
            raise ValueError(
                "tp_size must be divisible by dp_size * cp_size: "
                f"{self.tp_size} % ({self.dp_size} * {self.cp_size}) != 0"
            )
        return self.tp_size // divisor

    @property
    def attn_dp_size(self) -> int:
        return self.dp_size

    @property
    def moe_tp_size(self) -> int:
        if self.tp_size % self.ep_size != 0:
            raise ValueError(
                "tp_size must be divisible by ep_size: "
                f"{self.tp_size} % {self.ep_size} != 0"
            )
        return self.tp_size // self.ep_size

    @property
    def moe_ep_size(self) -> int:
        return self.ep_size


class SimulationMode(Enum):
    BLOCKING = "BLOCKING"
    OFFLINE = "OFFLINE"


@dataclass(slots=True)
class RequestStats:
    rid: str = ""
    last_event_time: float = 0.0
    input_length: int = 1
    output_length: int = 1

    # Prefix cache stats
    recv_device_hit_len: int = 0
    # Device hit length before `get_new_batch_prefill`.
    # It may decrease if queued requests trigger KV eviction.
    before_adder_device_hit_len: int = 0
    final_device_hit_len: int = 0
    recv_host_hit_len: int = 0  # Host hit length before prefetch
    final_host_hit_len: int = 0  # Host hit length after prefetch
    recv_storage_hit_len: int = 0  # Storage hit length at prefetch enqueue
    final_storage_hit_len: int = 0  # Storage hit length at prefetch end

    queue_start: float = -1
    queue_end: float = -1
    created_time: float = -1
    gen_token_latencies: list[float] = field(default_factory=list)

    def is_complete(self) -> bool:
        return True


def _bandwidth_property(gb_attr: str):
    def getter(self):
        gb_value = getattr(self, gb_attr)
        return gb_value * 1e9 if gb_value else None

    return property(getter)


@dataclass
class PlatformConfig:
    device: Union[AcceleratorInfo, str]
    # Storage configuration for hierarchical cache management.
    disk_capacity_gb: Optional[float] = None
    disk_read_bandwidth_gb: Optional[float] = None
    disk_write_bandwidth_gb: Optional[float] = None
    memory_capacity_gb: Optional[float] = None
    memory_read_bandwidth_gb: Optional[float] = None
    memory_write_bandwidth_gb: Optional[float] = None
    num_device_per_node: int = 8

    # Bandwidth properties (in bytes, converted from GB)
    disk_read_bandwidth = _bandwidth_property("disk_read_bandwidth_gb")
    disk_write_bandwidth = _bandwidth_property("disk_write_bandwidth_gb")
    memory_read_bandwidth = _bandwidth_property("memory_read_bandwidth_gb")
    memory_write_bandwidth = _bandwidth_property("memory_write_bandwidth_gb")
