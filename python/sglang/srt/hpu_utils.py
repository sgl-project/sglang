# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.utils import get_bool_env_var, get_int_env_var, is_hpu


@dataclass
class HPUBlockMetadata:
    """HPU-specific metadata for paged attention."""

    use_contiguous_pa: Optional[bool] = None
    block_list: Optional[torch.Tensor] = None
    block_mapping: Optional[torch.Tensor] = None
    block_groups: Optional[torch.Tensor] = None
    block_usage: Optional[torch.Tensor] = None

    def init_block_metadata(self, block_tables, slot_mapping, block_size, batch_size):
        """Initialize block metadata for HPU paged attention."""
        device = "cpu"

        # Calculate block metadata
        last_block_usage = [slot % block_size + 1 for slot in slot_mapping]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
        block_usage = [
            [block_size] * (len(bt) - 1) + [lbu]
            for bt, lbu in zip(block_tables, last_block_usage)
            if bt
        ]
        block_list = flatten(block_tables)
        block_groups = flatten(block_groups)
        block_usage = flatten(block_usage)
        assert len(block_list) == len(block_groups)
        assert len(block_list) == len(block_usage)

        if self.use_contiguous_pa:
            # Pad block metadata if needed
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            block_bucket_size = find_bucket(
                block_bucket_size,
                (
                    DECODE_BLOCK_BUCKET_MIN,
                    DECODE_BLOCK_BUCKET_STEP,
                    DECODE_BLOCK_BUCKET_MAX,
                ),
            )
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(
                tensor, indices, pad_value
            )
        else:
            block_bucket_size = find_bucket(
                len(block_list),
                (
                    DECODE_BLOCK_BUCKET_MIN,
                    DECODE_BLOCK_BUCKET_STEP,
                    DECODE_BLOCK_BUCKET_MAX,
                ),
            )
            padding_fn = lambda tensor, pad_value: pad_list(
                tensor, block_bucket_size, pad_value
            )

        block_list = padding_fn(block_list, _PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, _PAD_BLOCK_GROUP)
        block_usage = padding_fn(block_usage, _PAD_BLOCK_USAGE)

        # Convert to tensors
        self.block_list = torch.tensor(block_list, dtype=torch.long, device=device)
        self.block_groups = torch.tensor(block_groups, dtype=torch.long, device=device)
        self.block_usage = torch.tensor(
            block_usage, dtype=torch.bfloat16, device=device
        )

        # Set block mapping and scales
        self.block_mapping, self.block_groups = _set_block_mapping(
            self, batch_size, device
        )


_is_hpu = is_hpu()

if _is_hpu:
    _PAD_BLOCK_ID = 0
    _PAD_BLOCK_USAGE = 1
    _PAD_BLOCK_GROUP = -1

    PREFILL_BUCKET_MIN = get_int_env_var("SGLANG_HPU_PREFILL_BUCKET_MIN", 1024)
    PREFILL_BUCKET_STEP = get_int_env_var("SGLANG_HPU_PREFILL_BUCKET_STEP", 1024)
    PREFILL_BUCKET_MAX = get_int_env_var("SGLANG_HPU_PREFILL_BUCKET_MAX", 5120)

    DECODE_BLOCK_BUCKET_MIN = get_int_env_var("SGLANG_HPU_DECODE_BLOCK_BUCKET_MIN", 128)
    DECODE_BLOCK_BUCKET_STEP = get_int_env_var(
        "SGLANG_HPU_DECODE_BLOCK_BUCKET_STEP", 128
    )
    DECODE_BLOCK_BUCKET_MAX = get_int_env_var(
        "SGLANG_HPU_DECODE_BLOCK_BUCKET_MAX", 2560
    )
    DECODE_BATCH_BUCKET_MIN = get_int_env_var("SGLANG_HPU_DECODE_BATCH_BUCKET_MIN", 1)
    DECODE_BATCH_BUCKET_STEP = get_int_env_var(
        "SGLANG_HPU_DECODE_BATCH_BUCKET_STEP", 32
    )
    DECODE_BATCH_BUCKET_MAX = get_int_env_var("SGLANG_HPU_DECODE_BATCH_BUCKET_MAX", 128)

    USE_CONTIGUOUS_PA = get_bool_env_var("SGLANG_HPU_USE_CONTIGUOUS_PA", "true")
    SKIP_WARMUP = get_bool_env_var("SGLANG_HPU_SKIP_WARMUP", "false")

    from vllm.utils import make_tensor_with_pad
    from vllm_hpu_extension.bucketing import find_bucket

    def get_prefill_all_seq_len_buckets():
        return list(
            range(PREFILL_BUCKET_MIN, PREFILL_BUCKET_MAX + 1, PREFILL_BUCKET_STEP)
        )

    def get_decode_all_batch_buckets():
        buckets = []
        batch = DECODE_BATCH_BUCKET_MIN
        if batch < DECODE_BATCH_BUCKET_STEP:
            while batch < DECODE_BATCH_BUCKET_STEP:
                buckets.append(batch)
                batch = batch * 2
        buckets.extend(
            range(
                DECODE_BATCH_BUCKET_STEP,
                DECODE_BATCH_BUCKET_MAX + 1,
                DECODE_BATCH_BUCKET_STEP,
            )
        )
        return buckets

    def get_decode_all_seq_len_buckets():
        return list(
            range(
                DECODE_BLOCK_BUCKET_MIN,
                DECODE_BLOCK_BUCKET_MAX + 1,
                DECODE_BLOCK_BUCKET_STEP,
            )
        )

    def get_decode_all_buckets():
        buckets = []
        for batch_size in get_decode_all_batch_buckets():
            for seq_len in get_decode_all_seq_len_buckets():
                if seq_len == DECODE_BLOCK_BUCKET_MIN:
                    buckets.append((batch_size, seq_len))
                elif (
                    seq_len // batch_size
                    > DECODE_BLOCK_BUCKET_MAX // DECODE_BATCH_BUCKET_MAX
                ):
                    continue
                else:
                    buckets.append((batch_size, seq_len))
        return buckets

    def flatten(in_list):
        return list(itertools.chain(*in_list))

    def gather_list(tensor, indices, pad_value):
        result = [pad_value] * len(indices)
        for i, idx in enumerate(indices):
            if idx is not None:
                result[i] = tensor[idx]
        return result

    def round_up(value: int, k: int) -> int:
        return (value + k - 1) // k * k

    def pad_list(input, k, v):
        input_len = len(input)
        target_len = round_up(input_len, k)
        padding = target_len - input_len
        return input + [v] * padding

    def _set_block_mapping(metadata: HPUBlockMetadata, batch_size, device):
        """Set block mapping using one-hot encoding of block groups."""
        # Handle out of bounds classes on CPU
        block_groups = metadata.block_groups.to(torch.long)
        block_mapping = torch.nn.functional.relu(block_groups)
        block_mapping = torch.nn.functional.one_hot(
            block_mapping, num_classes=batch_size
        )
        oob_values = block_groups.lt(0)
        block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
        block_groups.masked_fill_(oob_values, batch_size)
        return block_mapping.to(torch.bfloat16), block_groups

    def get_prefill_seq_len_bucket(sum_seq_len):
        return find_bucket(
            sum_seq_len, (PREFILL_BUCKET_MIN, PREFILL_BUCKET_STEP, PREFILL_BUCKET_MAX)
        )

    def get_decode_batch_bucket(batch_size):
        return find_bucket(
            batch_size,
            (
                DECODE_BATCH_BUCKET_MIN,
                DECODE_BATCH_BUCKET_STEP,
                DECODE_BATCH_BUCKET_MAX,
            ),
        )

    def create_hpu_block_metadata(worker_batch, page_size, req_token_pool):
        hpu_metadata = HPUBlockMetadata()
        if worker_batch.forward_mode.is_decode():
            batch_size = len(worker_batch.seq_lens)
            seq_len_list = worker_batch.seq_lens.tolist()
            padded_batch_size = get_decode_batch_bucket(batch_size)
            block_tables = []
            slots_list = []
            for i in range(batch_size):
                num_pages = (seq_len_list[i] + page_size - 1) // page_size
                num_lots_aligned = num_pages * page_size
                slots = req_token_pool.req_to_token[
                    worker_batch.req_pool_indices[i], :num_lots_aligned
                ]
                pages = (slots // page_size).view(-1, page_size)[:, 0]
                block_tables.append(pages.flatten().tolist())
                slots_list.append(slots)
            for i in range(padded_batch_size - batch_size):
                block_tables.append([_PAD_BLOCK_ID])

            padding_len = padded_batch_size - len(worker_batch.seq_lens)
            slot_mapping = torch.nn.functional.pad(
                worker_batch.out_cache_loc, (0, padding_len), value=0
            )

            # Create HPUBlockMetadata instance
            hpu_metadata = HPUBlockMetadata(use_contiguous_pa=USE_CONTIGUOUS_PA)
            hpu_metadata.init_block_metadata(
                block_tables, slot_mapping, page_size, padded_batch_size
            )
        return hpu_metadata

    def make_cpu_tensor(data, max_len, pad, dtype, flat):
        if flat:
            data = [flatten(data)]
        result = make_tensor_with_pad(
            data, max_len=max_len, pad=pad, dtype=dtype, device="cpu"
        )
        return result

    def prepare_hpu_attn_bias_prefill(seq_lens, max_prompt_len, dtype):
        seq_pos = [list(range(sl)) for sl in seq_lens]
        seq_idx = [[i] * sl for i, sl in enumerate(seq_lens)]
        seq_pos = make_cpu_tensor(
            seq_pos, max_len=max_prompt_len, pad=-1, dtype=torch.long, flat=True
        )
        seq_idx = make_cpu_tensor(
            seq_idx, max_len=max_prompt_len, pad=-1, dtype=torch.long, flat=True
        )
        attn_bias = torch.zeros(1, 1, max_prompt_len, max_prompt_len, dtype=dtype)
        return attn_bias, seq_pos, seq_idx

    def compute_hpu_attn_bias_prefill(seq_pos, seq_idx, dtype):
        q_seq_idx = seq_idx.unsqueeze(-1)
        kv_seq_idx = seq_idx.unsqueeze(-2)
        q_seq_pos = seq_pos.unsqueeze(-1)
        kv_seq_pos = seq_pos.unsqueeze(-2)
        seq_idx = q_seq_idx != kv_seq_idx
        seq_pos = kv_seq_pos > q_seq_pos
        attn_mask = seq_idx | seq_pos
        attn_bias = torch.zeros_like(attn_mask, dtype=dtype)
        attn_bias.masked_fill_(attn_mask, -math.inf)
        return attn_bias.unsqueeze(1)

    def to_hpu_and_pad_1d(tensor, pad_len, pad_value=0):
        return torch.nn.functional.pad(tensor.to("hpu"), (0, pad_len), value=pad_value)

    def compute_hpu_attn_bias_decode(page_size, block_usage, dtype):
        mask = torch.arange(0, page_size, device="hpu", dtype=torch.int32).unsqueeze(0)
        mask = mask >= block_usage.to("hpu").unsqueeze(-1)
        attn_bias = (
            torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf).clone()
        )
        return attn_bias
