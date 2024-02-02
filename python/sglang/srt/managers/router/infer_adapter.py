import logging
from dataclasses import dataclass
from typing import List, Dict

import torch

from sglang.srt.memory_pool import TokenToKVPool
from sglang.srt.models.lora import get_mapped_params

logger = logging.getLogger("infer_adapter")


@dataclass
class InferAdapter:
    adapter_uids: List[str] # list of active adapters
    lora_idx: Dict[str, int] # adapter uid -> index in adapter_uids
    token_to_kv_pool: TokenToKVPool
    a_loc: torch.Tensor  # a_loc[i] is a list of indices occupied by adapter i
    a_start: torch.Tensor  # a_start[i] is the start location of adapter i
    a_len: torch.Tensor  # a_len[i] is the number of cells occupied by adapter i
    a_scaling: torch.Tensor  # a_scaling[i] is the scaling factor of adapter i

    @classmethod
    def init(cls, token_to_kv_pool):
        return cls(
            adapter_uids=[],
            lora_idx={},
            token_to_kv_pool=token_to_kv_pool,
            a_loc=torch.empty(0, dtype=torch.long, device="cuda"),
            a_start=torch.empty(0, dtype=torch.long, device="cuda"),
            a_len=torch.empty(0, dtype=torch.long, device="cuda"),
            a_scaling=torch.empty(0, dtype=torch.float16, device="cuda"),
        )

    def add_zero_lora(self):
        pass


    def load_lora(self, adapter, loc):
        for i in range(adapter.base_config.num_hidden_layers):
            adapter.layers[i].load_to_gpu(mode="paged")
            w_combined = adapter.layers[i].w_combined
            self.token_to_kv_pool.kv_data[i][loc] = w_combined
            adapter.layers[i].offload_from_gpu(mode="paged")


    def load(self, adapters):
        if len(adapters) == 0:
            logger.info(f"load 0 adapters, {len(self.adapter_uids)} in total")
            return

        new_adapters = []
        tot_size = 0
        for adapter in adapters:
            if adapter is not None and adapter.uid not in self.lora_idx:
                new_adapters.append(adapter)
                tot_size += adapter.r * len(adapter.paged_modules)
        logger.info(f"load {len(new_adapters)} adapters, {len(self.adapter_uids) + len(new_adapters)} in total")

        new_loc = self.token_to_kv_pool.alloc(tot_size)
        assert new_loc is not None, "no space for new adapters"
        start_offset = self.a_start.shape[0]
        self.a_start = torch.cat((self.a_start, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        len_offset = self.a_len.shape[0]
        self.a_len = torch.cat((self.a_len, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        loc_offset = self.a_loc.shape[0]
        self.a_loc = torch.cat((self.a_loc, torch.empty(tot_size, dtype=torch.long, device="cuda"))) 

        cum_loc = 0
        cum_loc_list = []
        for i, new_adapter in enumerate(new_adapters):
            cum_loc_list.append(cum_loc)
            self.lora_idx[new_adapter.uid] = len(self.adapter_uids)
            self.adapter_uids.append(new_adapter.uid)
            self.a_start[start_offset + i] = loc_offset + cum_loc
            num_loc = new_adapter.r * len(new_adapter.paged_modules)
            self.a_len[len_offset + i] = num_loc
            self.a_loc[loc_offset + cum_loc: loc_offset + cum_loc + num_loc] = (
                    new_loc[cum_loc: cum_loc + num_loc])
            cum_loc += num_loc
        self.a_scaling = torch.cat((self.a_scaling, torch.tensor([adapter.scaling for adapter in new_adapters], dtype=torch.float16, device="cuda")))

        for i, new_adapter in enumerate(new_adapters):
            cum_loc = cum_loc_list[i]
            self.load_lora(new_adapter, new_loc[cum_loc: cum_loc + self.a_len[len_offset + i]])
