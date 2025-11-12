import atexit
import signal
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from sglang.srt.utils import get_bool_env_var, get_int_env_var

use_elasticmem = get_bool_env_var("SGLANG_ELASTIC_MEM_POOL", "true")
# use_elasticmem = get_bool_env_var("SGLANG_ELASTIC_MEM_POOL", "false")
cu_page_size = get_int_env_var("SGLANG_CU_PAGE_SIZE", 2 << 20)

if use_elasticmem:
    import kvcached.vmm_ops as vmm_ops


class ElasticMempool(ABC):
    @abstractmethod
    def create_elastic_buffers(self):
        pass

    @abstractmethod
    def disable(self, indices: List[int]) -> Tuple[int, List[int], List[int]]:
        pass

    @abstractmethod
    def enable(self, indices: List[int]) -> Tuple[int, List[int], List[int]]:
        pass


class ElasticMempoolOrchestrator:
    def __init__(self):
        assert use_elasticmem
        atexit.register(vmm_ops.shutdown_emem)
        signal.signal(signal.SIGINT, lambda sig, frame: vmm_ops.shutdown_emem())
        signal.signal(signal.SIGTERM, lambda sig, frame: vmm_ops.shutdown_emem())
        signal.signal(signal.SIGQUIT, lambda sig, frame: vmm_ops.shutdown_emem())

        current_device_id = f"cuda:{torch.cuda.current_device()}"
        vmm_ops.init_emem(current_device_id, cu_page_size)

        self.elastic_mempools = []

    def register_pool(self, mem_pool: ElasticMempool):
        self.elastic_mempools.append(mem_pool)
