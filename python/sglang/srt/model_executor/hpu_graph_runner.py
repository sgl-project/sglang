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
"""Run the model with hpu graph."""

from __future__ import annotations

import logging
import math
import os
import time
from collections import namedtuple
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
import tqdm

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.utils import is_hpu

_is_hpu = is_hpu()
if _is_hpu:

    if torch.hpu.is_lazy():
        os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"

    from sglang.srt.hpu_utils import (
        SKIP_WARMUP,
        USE_CONTIGUOUS_PA,
        compute_hpu_attn_bias_decode,
        compute_hpu_attn_bias_prefill,
        get_decode_all_buckets,
        get_decode_batch_bucket,
        get_prefill_all_seq_len_buckets,
        get_prefill_seq_len_bucket,
        prepare_hpu_attn_bias_prefill,
        to_hpu_and_pad_1d,
    )

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

HPUForwardBatch = namedtuple(
    "HPUForwardBatch",
    [
        "forward_mode",
        "batch_size",
        "input_ids",
        "out_cache_loc",
        "positions",
        "attn_bias",
        "seq_pos",
        "seq_idx",
        "valid_seq_len",
        "extend_seq_lens",
        "page_size",
        "block_list",
        "block_mapping",
        "block_groups",
        "block_usage",
        "attn_backend",
        "token_to_kv_pool",
        "use_contiguous_pa",
        "input_embeds",
        "extend_return_logprob",
        "padded_static_len",
        "capture_hidden_mode",
    ],
    defaults=[None, False, -1, CaptureHiddenMode.NULL],
)


def create_hpu_forward_batch(forward_batch: ForwardBatch, model_runner: ModelRunner):
    assert (
        forward_batch.hpu_metadata is not None
    ), "Expected HPU Metadata for HPU forward batch"
    batch_size = forward_batch.batch_size
    page_size = model_runner.token_to_kv_pool_allocator.page_size
    if forward_batch.forward_mode.is_extend():
        seq_len_list = forward_batch.extend_seq_lens
        sum_seq_len = seq_len_list.sum()
        max_prompt_len = get_prefill_seq_len_bucket(sum_seq_len)
        attn_bias, seq_pos, seq_idx = prepare_hpu_attn_bias_prefill(
            seq_lens=seq_len_list,
            max_prompt_len=max_prompt_len,
            dtype=model_runner.dtype,
        )
        attn_bias = attn_bias.to("hpu")
        seq_pos = seq_pos.to("hpu")
        seq_idx = seq_idx.to("hpu")
        padding_len = max_prompt_len - sum_seq_len
        max_prefill_seqs = model_runner.server_args.max_running_requests
        input_ids = to_hpu_and_pad_1d(forward_batch.input_ids, padding_len)
        positions = to_hpu_and_pad_1d(forward_batch.positions, padding_len)
        valid_seq_len = sum_seq_len.to("hpu", dtype=torch.int64)
        extend_seq_lens_padded = to_hpu_and_pad_1d(
            forward_batch.extend_seq_lens, max_prefill_seqs - batch_size
        )
        out_cache_loc = to_hpu_and_pad_1d(forward_batch.out_cache_loc, padding_len)
        batch_size = 1
        block_list = None
        block_mapping = None
        block_groups = None
        block_usage = None
        use_contiguous_pa = None
    else:
        padded_batch_size = get_decode_batch_bucket(batch_size)
        padding_len = padded_batch_size - batch_size
        input_ids = to_hpu_and_pad_1d(
            forward_batch.input_ids.to(torch.int64), padding_len
        )
        positions = to_hpu_and_pad_1d(
            forward_batch.positions.to(torch.int64), padding_len
        )
        valid_seq_len = torch.ones(padded_batch_size, dtype=torch.int64, device="hpu")
        out_cache_loc = to_hpu_and_pad_1d(forward_batch.out_cache_loc, padding_len)
        batch_size = padded_batch_size
        attn_bias = compute_hpu_attn_bias_decode(
            page_size, forward_batch.hpu_metadata.block_usage, model_runner.dtype
        )

        seq_pos = None
        seq_idx = None
        extend_seq_lens_padded = None
        block_list = forward_batch.hpu_metadata.block_list.to("hpu")
        block_mapping = forward_batch.hpu_metadata.block_mapping.to("hpu")
        block_groups = forward_batch.hpu_metadata.block_groups.to("hpu")
        block_usage = forward_batch.hpu_metadata.block_usage.to("hpu")
        use_contiguous_pa = forward_batch.hpu_metadata.use_contiguous_pa

    return HPUForwardBatch(
        forward_mode=forward_batch.forward_mode,
        batch_size=batch_size,
        input_ids=input_ids,
        out_cache_loc=out_cache_loc,
        positions=positions,
        attn_bias=attn_bias,
        seq_pos=seq_pos,
        seq_idx=seq_idx,
        valid_seq_len=valid_seq_len,
        extend_seq_lens=extend_seq_lens_padded,
        page_size=page_size,
        block_list=block_list,
        block_mapping=block_mapping,
        block_groups=block_groups,
        block_usage=block_usage,
        attn_backend=forward_batch.attn_backend,
        token_to_kv_pool=forward_batch.token_to_kv_pool,
        use_contiguous_pa=use_contiguous_pa,
    )


def create_hpu_dummy_batch_prefill(
    seq_len, dtype, page_size, max_running_requests, attn_backend, token_to_kv_pool
):
    return HPUForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.zeros(seq_len, dtype=torch.int64, device="hpu"),
        out_cache_loc=torch.arange(seq_len, dtype=torch.int64, device="hpu"),
        positions=torch.zeros(seq_len, dtype=torch.int64, device="hpu"),
        attn_bias=torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device="hpu"),
        seq_pos=torch.zeros(1, seq_len, dtype=torch.int64, device="hpu"),
        seq_idx=torch.zeros(1, seq_len, dtype=torch.int64, device="hpu"),
        valid_seq_len=torch.ones((), dtype=torch.int64, device="hpu"),
        extend_seq_lens=torch.ones(
            max_running_requests,
            dtype=torch.int32,
            device="hpu",
        ),
        page_size=page_size,
        block_list=None,
        block_mapping=None,
        block_groups=None,
        block_usage=None,
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        use_contiguous_pa=None,
    )


def create_hpu_dummy_batch_decode(
    batch_size, block_num, dtype, page_size, attn_backend, token_to_kv_pool
):
    return HPUForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=torch.zeros(batch_size, dtype=torch.int64, device="hpu"),
        out_cache_loc=torch.zeros(batch_size, dtype=torch.int64, device="hpu"),
        positions=torch.zeros(batch_size, dtype=torch.int64, device="hpu"),
        attn_bias=torch.zeros(block_num, page_size, dtype=dtype, device="hpu"),
        seq_pos=None,
        seq_idx=None,
        valid_seq_len=torch.ones(batch_size, dtype=torch.int64, device="hpu"),
        extend_seq_lens=None,
        page_size=page_size,
        block_list=torch.zeros(block_num, dtype=torch.int64, device="hpu"),
        block_mapping=torch.zeros(
            block_num, batch_size, dtype=torch.bfloat16, device="hpu"
        ),
        block_groups=torch.zeros(block_num, dtype=torch.int64, device="hpu"),
        block_usage=torch.zeros(block_num, dtype=torch.bfloat16, device="hpu"),
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        use_contiguous_pa=USE_CONTIGUOUS_PA,
    )


class HPUAdapter:

    def __init__(self, model, dtype) -> None:
        self.model = model
        self.dtype = dtype

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        assert len(args) == 3, "Only three arguments are supported"
        input_batch = args[2]
        if input_batch.forward_mode.is_extend():
            input_batch.attn_bias.copy_(
                compute_hpu_attn_bias_prefill(
                    input_batch.seq_pos, input_batch.seq_idx, self.dtype
                )
            )
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class HPUGraphRunner:
    """A HPUGraphRunner runs the forward pass of a model with HPU graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        import habana_frameworks.torch as htorch
        import vllm_hpu_extension.environment as environment

        environment.runtime_params["model_type"] = (
            model_runner.model_config.hf_config.model_type
        )

        self.model = (
            htorch.hpu.wrap_in_hpu_graph(
                HPUAdapter(self.model_runner.model, self.model_runner.dtype),
                disable_tensor_cache=True,
            )
            if htorch.utils.internal.is_lazy()
            else HPUAdapter(self.model_runner.model, self.model_runner.dtype)
        )
        # Capture
        if not SKIP_WARMUP:
            try:
                with self.model_capture_mode():
                    logger.info(
                        "Begin to capture hpu graph, you can use `export SGLANG_HPU_SKIP_WARMUP=true` to skip this step."
                    )
                    time_start = time.perf_counter()
                    self.capture()
                    time_end = time.perf_counter()
                    logger.info(
                        f"Capture hpu graph time: {time_end - time_start} seconds"
                    )
                    logger.info("Capture hpu graph success")
            except RuntimeError as e:
                raise Exception(f"Capture hpu graph failed: {e}\n")

    @contextmanager
    def model_capture_mode(self):
        yield

    def can_run(self, forward_batch: ForwardBatch):
        return True

    def capture(self):
        # prefill
        time_start = time.perf_counter()
        prefill_seq_len_buckets = get_prefill_all_seq_len_buckets()
        for seq_len in prefill_seq_len_buckets:
            self.capture_prefill(seq_len)
        time_end = time.perf_counter()
        logger.info(f"Capture prefill time: {time_end - time_start} seconds")

        # decode
        time_start = time.perf_counter()
        all_buckets = get_decode_all_buckets()
        for batch_size, seq_len in all_buckets:
            self.capture_decode(batch_size, seq_len)
        time_end = time.perf_counter()
        logger.info(f"Capture decode time: {time_end - time_start} seconds")

    def capture_prefill(self, seq_len):
        logger.info(f"Capture prefill with seq_len: {seq_len}")
        forward_batch = create_hpu_dummy_batch_prefill(
            seq_len,
            self.model_runner.dtype,
            self.model_runner.token_to_kv_pool_allocator.page_size,
            self.model_runner.server_args.max_running_requests,
            self.model_runner.attn_backend,
            self.model_runner.token_to_kv_pool,
        )
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        for i in range(3):
            self.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )

    def capture_decode(self, batch_size, block_num):
        logger.info(
            f"Capture decode with batch_size: {batch_size} and block_num: {block_num}"
        )
        page_size = self.model_runner.token_to_kv_pool_allocator.page_size
        forward_batch = create_hpu_dummy_batch_decode(
            batch_size,
            block_num,
            self.model_runner.dtype,
            page_size,
            self.model_runner.attn_backend,
            self.model_runner.token_to_kv_pool,
        )
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        for i in range(3):
            self.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )

    def _forward(self, forward_batch: ForwardBatch):
        import habana_frameworks.torch as htorch

        forward_batch_hpu = create_hpu_forward_batch(forward_batch, self.model_runner)
        results = self.model.forward(
            forward_batch_hpu.input_ids, forward_batch_hpu.positions, forward_batch_hpu
        )
        htorch.core.mark_step()
        logits_output = LogitsProcessorOutput(
            next_token_logits=results.next_token_logits.clone()[
                : forward_batch.batch_size
            ],
            hidden_states=(
                results.hidden_states.clone()[: forward_batch.batch_size]
                if results.hidden_states is not None
                else None
            ),
        )
        return logits_output

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        return self._forward(forward_batch)
