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
    mem_fraction_static: Optional[float] = None
    max_total_tokens: Optional[int] = None

    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1

    # framework backend
    backend_name: str = "sglang"
    backend_version: Optional[str] = None

    @property
    def attn_tp_size(self) -> int:
        return self.tp_size / self.dp_size

    @property
    def attn_dp_size(self) -> int:
        return self.dp_size

    @property
    def moe_tp_size(self) -> int:
        return self.tp_size / self.ep_size

    @property
    def moe_ep_size(self) -> int:
        return self.ep_size


class SimulationMode(Enum):
    BLOCKING = "BLOCKING"
    OFFLINE = "OFFLINE"


@dataclass
class RequestStats:
    rid: str = ""
    last_event_time: float = 0.0
    input_length: int = 1
    output_length: int = 1
    final_reused_tokens: int = 0
    prefetch_complete_tokens: int = 0
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
