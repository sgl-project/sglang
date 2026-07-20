from __future__ import annotations

from itertools import chain
from typing import Optional

import torch

from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.common import is_pin_memory_available


class ScheduleBatchDllmMixin:
    def prepare_for_dllm_block_extend(self, buffers: Optional[dict] = None):
        self.forward_mode = ForwardMode.DLLM_EXTEND

        reqs = self.reqs
        bs = len(reqs)
        device = self.device

        fill_ids = [r.get_fill_ids() for r in reqs]
        input_ids = [ids[len(r.prefix_indices) :] for ids, r in zip(fill_ids, reqs)]
        seq_lens = [len(ids) for ids in fill_ids]
        orig_seq_lens = [
            max(len(r.full_untruncated_fill_ids), len(r.origin_input_ids)) for r in reqs
        ]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_range.length for r in reqs]
        input_id_lens = [len(ids) for ids in input_ids]
        extend_num_tokens = sum(input_id_lens)

        flat_input_ids = list(chain.from_iterable(input_ids))

        cache = buffers
        if (
            cache is None
            or cache["bs"] != bs
            or cache["extend_num_tokens"] < extend_num_tokens
        ):
            pin_memory = is_pin_memory_available(device)
            cache = {
                "bs": bs,
                "extend_num_tokens": extend_num_tokens,
                "input_ids_host": torch.empty(
                    extend_num_tokens, dtype=torch.int64, pin_memory=pin_memory
                ),
                "input_ids_dev": torch.empty(
                    extend_num_tokens, dtype=torch.int64, device=device
                ),
                "seq_lens_host": torch.empty(
                    bs, dtype=torch.int64, pin_memory=pin_memory
                ),
                "seq_lens_dev": torch.empty(bs, dtype=torch.int64, device=device),
                "seq_lens_cpu": torch.empty(bs, dtype=torch.int64),
                "orig_seq_lens_host": torch.empty(
                    bs, dtype=torch.int32, pin_memory=pin_memory
                ),
                "orig_seq_lens_dev": torch.empty(bs, dtype=torch.int32, device=device),
            }
        self._dllm_block_buffers = cache

        cache["input_ids_host"][:extend_num_tokens].copy_(
            torch.tensor(flat_input_ids, dtype=torch.int64), non_blocking=False
        )
        cache["seq_lens_host"][:bs].copy_(
            torch.tensor(seq_lens, dtype=torch.int64), non_blocking=False
        )
        cache["seq_lens_cpu"][:bs] = cache["seq_lens_host"][:bs]
        cache["orig_seq_lens_host"][:bs].copy_(
            torch.tensor(orig_seq_lens, dtype=torch.int32), non_blocking=False
        )

        cache["input_ids_dev"][:extend_num_tokens].copy_(
            cache["input_ids_host"][:extend_num_tokens], non_blocking=True
        )
        cache["seq_lens_dev"][:bs].copy_(cache["seq_lens_host"][:bs], non_blocking=True)
        cache["orig_seq_lens_dev"][:bs].copy_(
            cache["orig_seq_lens_host"][:bs], non_blocking=True
        )

        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = cache["seq_lens_dev"][:bs]
        self.seq_lens_cpu = cache["seq_lens_cpu"][:bs].clone()
        self.extend_num_tokens = extend_num_tokens

        out_cache_loc, req_pool_indices_tensor, req_pool_indices = alloc_for_extend(
            self
        )

        for i, (req, seq_len) in enumerate(zip(reqs, seq_lens)):
            req.req_pool_idx = req_pool_indices[i]
            req.extend_batch_idx += 1
            req.kv_committed_len = seq_len
            req.kv.kv_allocated_len = seq_len

        self.input_ids = cache["input_ids_dev"][:extend_num_tokens]
        self.req_pool_indices = req_pool_indices_tensor
        self.orig_seq_lens = cache["orig_seq_lens_dev"][:bs]
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)
