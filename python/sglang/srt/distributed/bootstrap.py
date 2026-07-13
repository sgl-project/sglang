import logging

import msgspec

from sglang.srt.utils import cpu_has_amx_support, is_host_cpu_arm64

logger = logging.getLogger(__name__)

_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()


class TorchDistributedResult(msgspec.Struct, frozen=True, kw_only=True):
    tp_group: object
    pp_group: object
    attention_tp_group: object
    pre_model_load_memory: float
