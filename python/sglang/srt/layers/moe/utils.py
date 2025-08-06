import importlib.util
from enum import Enum
from functools import lru_cache

from packaging import version as pkg_version

from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.layers.dp_attention import get_attention_dp_size
from sglang.srt.managers.schedule_batch import global_server_args_dict


@lru_cache(maxsize=1)
def should_use_flashinfer_trtllm_moe():
    result = global_server_args_dict["enable_flashinfer_trtllm_moe"] and (
        not importlib.util.find_spec("flashinfer")
        or pkg_version.parse(__import__("flashinfer").__version__)
        >= pkg_version.parse("0.2.9rc1")
    )
    return result


def should_use_flashinfer_cutlass_moe_fp4_allgather():
    """
    Perform FP4 quantize before all-gather for flashinfer cutlass moe to reduce communication cost for high-throughput serving.
    """
    return (
        not global_server_args_dict["disable_flashinfer_cutlass_moe_fp4_allgather"]
        and global_server_args_dict["enable_flashinfer_cutlass_moe"]
        and global_server_args_dict["enable_dp_attention"]
        and get_moe_expert_parallel_world_size() == get_attention_dp_size()
    )


class MoeA2ABackend(Enum):

    STANDARD = ("standard", "none")
    DEEPEP = "deepep"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.STANDARD
        for member in cls:
            if value in member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")

    def is_deepep(self):
        return self == MoeA2ABackend.DEEPEP

    def is_standard(self):
        return self == MoeA2ABackend.STANDARD


class DeepEPMode(Enum):
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self):
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self):
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool):
        if self != DeepEPMode.AUTO:
            return self

        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY
