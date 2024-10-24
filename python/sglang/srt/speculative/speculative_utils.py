from __future__ import annotations

from typing import TYPE_CHECKING, List, Type

import torch
import triton
import triton.language as tl

from .build_eagle_tree import build_tree_kernel
from sglang.srt.model_executor.forward_batch_info import ForwardMode, ForwardBatch

if TYPE_CHECKING:
    from python.sglang.srt.layers.sampler import SampleOutput
    from python.sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs


class SpecInput:
    pass

class SpecVerifyInput(SpecInput):
    pass

class SpecDraftInput(SpecInput):
    def prepare_for_extend(self, batch):
        raise NotImplementedError()

    def prepare_for_decode(self, batch):
        raise NotImplementedError()

    def generate_attn_arg(
        self,
        req_pool_indices: List,
        paged_kernel_lens: List,
        req_to_token_pool: ReqToTokenPool,
    ):
        raise NotImplementedError()

    def clear():
        pass
    
    def merge_batch(self, batch: SpecDraftInput):
        raise NotImplementedError()


class SpecInfoFactory:
    def __init__(self):
        self.factory = {}

    def register(self, alg_name: str, type_name: str) -> SpecInput:
        def wrapper(info: Type[SpecInput]) -> Type[SpecInput]:
            assert type_name in ['DraftInput', 'VerifyInput']
            if alg_name not in self.factory:
                self.factory[alg_name] = {}
            self.factory[alg_name].update({type_name: info})
            return info

        return wrapper

    def get(self, alg_name, type_name: str):
        if alg_name is None:
            return None
        return self.factory[alg_name][type_name]


DraftInfoFactory = SpecInfoFactory()


