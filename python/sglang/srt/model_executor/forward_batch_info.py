from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Meta data for a forward pass."""
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List

import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the first part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both PREFILL and EXTEND.
    MIXED = auto()

    def is_prefill(self):
        return self == ForwardMode.PREFILL

    def is_extend(self):
        return self == ForwardMode.EXTEND or self == ForwardMode.MIXED

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED


@dataclass
class InputMetadata:
    """Store all inforamtion of a forward pass."""

    forward_mode: ForwardMode
    sampling_info: SamplingBatchInfo
    batch_size: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: BaseTokenToKVPool
    attn_backend: AttentionBackend

    # Output location of the KV cache
    out_cache_loc: torch.Tensor

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_seq_lens: torch.Tensor = None
    extend_prefix_lens: torch.Tensor = None
    extend_start_loc: torch.Tensor = None
    extend_no_prefix: bool = None

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None
    extend_seq_lens_cpu: List[int] = None
    logprob_start_lens_cpu: List[int] = None

    # For multimodal
    pixel_values: List[torch.Tensor] = None
    image_sizes: List[List[List[int]]] = None
    image_offsets: List[List[int]] = None
    modalities: List[List[str]] = None

    def init_multimuldal_info(self, batch: ScheduleBatch):
        reqs = batch.reqs
        self.pixel_values = [r.pixel_values for r in reqs]
        self.image_sizes = [r.image_sizes for r in reqs]
        self.image_offsets = [r.image_offsets for r in reqs]
        self.modalities = [r.modalities for r in reqs]

    def compute_positions(self, batch: ScheduleBatch):
        position_ids_offsets = batch.position_ids_offsets

        if self.forward_mode.is_decode():
            if True:
                self.positions = self.seq_lens - 1
            else:
                # Deprecated
                self.positions = (self.seq_lens - 1) + position_ids_offsets
        else:
            if True:
                self.positions = torch.tensor(
                    np.concatenate(
                        [
                            np.arange(batch.prefix_lens_cpu[i], len(req.fill_ids))
                            for i, req in enumerate(batch.reqs)
                        ],
                        axis=0,
                    ),
                    device="cuda",
                )
            else:
                # Deprecated
                position_ids_offsets_cpu = position_ids_offsets.cpu().numpy()
                self.positions = torch.tensor(
                    np.concatenate(
                        [
                            np.arange(
                                batch.prefix_lens_cpu[i] + position_ids_offsets_cpu[i],
                                len(req.fill_ids) + position_ids_offsets_cpu[i],
                            )
                            for i, req in enumerate(batch.reqs)
                        ],
                        axis=0,
                    ),
                    device="cuda",
                )

        # Positions should be in long type
        self.positions = self.positions.to(torch.int64)

    def compute_extend_infos(self, batch: ScheduleBatch):
        extend_lens_cpu = [
            len(r.fill_ids) - batch.prefix_lens_cpu[i] for i, r in enumerate(batch.reqs)
        ]
        self.extend_seq_lens = torch.tensor(extend_lens_cpu, device="cuda")
        self.extend_prefix_lens = torch.tensor(batch.prefix_lens_cpu, device="cuda")
        self.extend_start_loc = torch.zeros_like(self.seq_lens)
        self.extend_start_loc[1:] = torch.cumsum(self.extend_seq_lens[:-1], dim=0)
        self.extend_no_prefix = all(l == 0 for l in batch.prefix_lens_cpu)

        self.extend_seq_lens_cpu = extend_lens_cpu
        self.logprob_start_lens_cpu = [
            (
                min(
                    req.logprob_start_len - batch.prefix_lens_cpu[i],
                    extend_lens_cpu[i] - 1,
                )
                if req.logprob_start_len >= batch.prefix_lens_cpu[i]
                else extend_lens_cpu[i] - 1  # Fake extend, actually decode
            )
            for i, req in enumerate(batch.reqs)
        ]

    @classmethod
    def from_schedule_batch(
        cls,
        model_runner: "ModelRunner",
        batch: ScheduleBatch,
    ):
        ret = cls(
            forward_mode=batch.forward_mode,
            sampling_info=batch.sampling_info,
            batch_size=batch.batch_size(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            attn_backend=model_runner.attn_backend,
            out_cache_loc=batch.out_cache_loc,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
        )

        ret.sampling_info.update_penalties()
        ret.sampling_info.update_regex_vocab_mask(batch)
        ret.compute_positions(batch)

        if not batch.forward_mode.is_decode():
            ret.init_multimuldal_info(batch)
            ret.compute_extend_infos(batch)

        model_runner.attn_backend.init_forward_metadata(batch, ret)

        return ret
