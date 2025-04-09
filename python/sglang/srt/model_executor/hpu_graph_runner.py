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
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations

import itertools
import math
import os
from collections import namedtuple
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import tqdm

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.utils import is_hpu

_is_hpu = is_hpu()
if _is_hpu:
    from vllm.utils import make_tensor_with_pad

    os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"

    from sglang.srt.hpu_utils import get_decode_batch_bucket, get_prefill_seq_len_bucket
if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


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
        "block_scales",
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


def flatten(in_list):
    return list(itertools.chain(*in_list))


def make_cpu_tensor(data, max_len, pad, dtype, flat):
    if flat:
        data = [flatten(data)]
    result = make_tensor_with_pad(
        data, max_len=max_len, pad=pad, dtype=dtype, device="cpu"
    )
    return result


def make_hpu_attn_bias(seq_lens, max_prompt_len, dtype):
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


def create_hpu_forward_batch(forward_batch: ForwardBatch, model_runner: ModelRunner):
    batch_size = forward_batch.batch_size
    page_size = model_runner.token_to_kv_pool_allocator.page_size
    if forward_batch.forward_mode.is_extend():
        seq_len_list = forward_batch.extend_seq_lens
        sum_seq_len = seq_len_list.sum()
        max_prompt_len = get_prefill_seq_len_bucket(sum_seq_len)
        attn_bias, seq_pos, seq_idx = make_hpu_attn_bias(
            seq_lens=seq_len_list,
            max_prompt_len=max_prompt_len,
            dtype=model_runner.dtype,
        )
        attn_bias = attn_bias.to("hpu")
        seq_pos = seq_pos.to("hpu")
        seq_idx = seq_idx.to("hpu")
        padding_len = max_prompt_len - sum_seq_len
        max_prefill_seqs = model_runner.server_args.max_running_requests
        input_ids = torch.nn.functional.pad(
            forward_batch.input_ids.to("hpu"), (0, padding_len), value=0
        )
        positions = torch.nn.functional.pad(
            forward_batch.positions.to("hpu"), (0, padding_len), value=0
        )
        valid_seq_len = sum_seq_len.to("hpu")
        extend_seq_lens_padded = torch.nn.functional.pad(
            forward_batch.extend_seq_lens.to("hpu"),
            (0, max_prefill_seqs - batch_size),
            value=0,
        )
        out_cache_loc = torch.nn.functional.pad(
            forward_batch.out_cache_loc.to("hpu"), (0, padding_len), value=0
        )
        batch_size = 1
        block_list = None
        block_mapping = None
        block_groups = None
        block_usage = None
        block_scales = None
    else:
        padded_batch_size = get_decode_batch_bucket(batch_size)
        padding_len = padded_batch_size - batch_size
        input_ids = torch.nn.functional.pad(
            forward_batch.input_ids.to("hpu"), (0, padding_len), value=0
        )
        positions = torch.nn.functional.pad(
            forward_batch.positions.to("hpu"), (0, padding_len), value=0
        )
        valid_seq_len = torch.ones(padded_batch_size, dtype=torch.int32, device="hpu")
        out_cache_loc = torch.nn.functional.pad(
            forward_batch.out_cache_loc.to("hpu"), (0, padding_len), value=0
        )
        batch_size = padded_batch_size
        mask = torch.arange(0, page_size, device="hpu", dtype=torch.int32).unsqueeze(0)
        mask = mask >= forward_batch.block_usage.to("hpu").unsqueeze(-1)
        attn_bias = torch.zeros_like(mask, dtype=model_runner.dtype).masked_fill_(
            mask, -math.inf
        )
        seq_pos = None
        seq_idx = None
        extend_seq_lens_padded = None
        block_list = forward_batch.block_list.to("hpu")
        block_mapping = forward_batch.block_mapping.to("hpu")
        block_groups = forward_batch.block_groups.to("hpu")
        block_usage = forward_batch.block_usage.to("hpu")
        block_scales = forward_batch.block_scales.to("hpu")

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
        block_scales=block_scales,
        attn_backend=forward_batch.attn_backend,
        token_to_kv_pool=forward_batch.token_to_kv_pool,
        use_contiguous_pa=forward_batch.use_contiguous_pa,
    )


def compute_hpu_attn_bias(seq_pos, seq_idx, dtype):
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
                compute_hpu_attn_bias(
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

        self.model = (
            htorch.hpu.wrap_in_hpu_graph(
                HPUAdapter(self.model_runner.model, self.model_runner.dtype),
                disable_tensor_cache=True,
            )
            if htorch.utils.internal.is_lazy()
            else HPUAdapter(self.model_runner.model, self.model_runner.dtype)
        )
        # Capture
        try:
            with self.model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(f"Capture hpu graph failed: {e}\n")

    @contextmanager
    def model_capture_mode(self):
        yield

    def can_run(self, forward_batch: ForwardBatch):
        return True

    def capture(self):
        # TODO: implement warmup
        pass

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
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        return self._forward(forward_batch)
