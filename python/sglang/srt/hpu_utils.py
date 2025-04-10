import itertools
import math
import os

import torch

from sglang.srt.utils import is_hpu

_is_hpu = is_hpu()

if _is_hpu:
    _PAD_BLOCK_ID = 0

    PREFILL_BUCKET_MIN = os.environ.get("SGLANG_HPU_PREFILL_BUCKET_MIN", 1024)
    PREFILL_BUCKET_STEP = os.environ.get("SGLANG_HPU_PREFILL_BUCKET_STEP", 1024)
    PREFILL_BUCKET_MAX = os.environ.get("SGLANG_HPU_PREFILL_BUCKET_MAX", 4096)

    DECODE_BLOCK_BUCKET_MIN = os.environ.get("SGLANG_HPU_DECODE_BLOCK_BUCKET_MIN", 128)
    DECODE_BLOCK_BUCKET_STEP = os.environ.get(
        "SGLANG_HPU_DECODE_BLOCK_BUCKET_STEP", 128
    )
    DECODE_BLOCK_BUCKET_MAX = os.environ.get("SGLANG_HPU_DECODE_BLOCK_BUCKET_MAX", 2560)
    DECODE_BATCH_BUCKET_MIN = os.environ.get("SGLANG_HPU_DECODE_BATCH_BUCKET_MIN", 1)
    DECODE_BATCH_BUCKET_STEP = os.environ.get("SGLANG_HPU_DECODE_BATCH_BUCKET_STEP", 32)
    DECODE_BATCH_BUCKET_MAX = os.environ.get("SGLANG_HPU_DECODE_BATCH_BUCKET_MAX", 128)

    USE_CONTIGUOUS_PA = (
        os.environ.get("SGLANG_HPU_USE_CONTIGUOUS_PA", "true").lower() == "true"
    )
    SKIP_WARMUP = os.environ.get("SGLANG_HPU_SKIP_WARMUP", "false").lower() == "true"

    from vllm_hpu_extension.bucketing import find_bucket
    from vllm_hpu_extension.ops import batch2block, block2batch

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

    def _set_block_mapping(metadata, batch_size, device):
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

    def _set_block_scales(metadata, device):
        """Set block scales using batch2block and block2batch operations."""
        block_mapping = metadata.block_mapping
        ones = torch.ones(
            (block_mapping.size(0),), device=device, dtype=block_mapping.dtype
        )
        sums = batch2block(block2batch(ones, block_mapping), block_mapping)
        block_scales = torch.reciprocal(torch.maximum(ones, sums))
        return block_scales

    def _init_block_metadata(ret, block_tables, slot_mapping, block_size, batch_size):
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

        if USE_CONTIGUOUS_PA:
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
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        # Convert to tensors
        ret.block_list = torch.tensor(block_list, dtype=torch.long, device=device)
        ret.block_groups = torch.tensor(block_groups, dtype=torch.long, device=device)
        ret.block_usage = torch.tensor(block_usage, dtype=torch.bfloat16, device=device)

        # Set block mapping and scales
        ret.block_mapping, ret.block_groups = _set_block_mapping(
            ret, batch_size, device
        )
        ret.block_scales = _set_block_scales(ret, device)

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

    def create_hpu_specific_fields(ret, page_size, req_token_pool):

        ret.page_size = page_size
        if ret.forward_mode.is_decode():
            ret.use_contiguous_pa = USE_CONTIGUOUS_PA
            batch_size = len(ret.seq_lens)
            padded_batch_size = get_decode_batch_bucket(batch_size)
            block_tables = []
            slots_list = []
            for i in range(batch_size):
                slots = req_token_pool.req_to_token[
                    ret.req_pool_indices[i], : ret.seq_lens[i]
                ]
                last_loc = slots[-1]
                num_full_tables = (ret.seq_lens[i] - 1) // page_size
                ranges = torch.arange(
                    0,
                    num_full_tables * page_size,
                    step=page_size,
                    device=ret.input_ids.device,
                )
                pages = slots[ranges] // page_size
                pages = pages.flatten().tolist()
                if last_loc % page_size != 0:
                    pages.append((last_loc // page_size).item())
                block_tables.append(pages)
                slots_list.append(slots)
            for i in range(padded_batch_size - batch_size):
                block_tables.append([_PAD_BLOCK_ID])

            padding_len = padded_batch_size - len(ret.seq_lens)
            slot_mapping = torch.nn.functional.pad(
                ret.out_cache_loc, (0, padding_len), value=0
            )
            _init_block_metadata(
                ret, block_tables, slot_mapping, page_size, padded_batch_size
            )
        return ret
