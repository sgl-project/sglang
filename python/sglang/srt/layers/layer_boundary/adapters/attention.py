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
"""The attention's inputs: the attention-TP context, its cached inputs, and
the attention-TP gather and slice."""

import logging
from contextlib import contextmanager
from typing import Callable

import torch

from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layer_boundary.layout import (
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_forward, get_parallel, get_spec
from sglang.srt.utils import is_cuda, is_npu

_is_cuda = is_cuda()
_is_npu = is_npu()


class AttentionInputs:
    def __init__(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        qkv_latent_func: Callable,
        *,
        is_pre_gathered: bool = False,
    ):
        self.hidden_states_local = hidden_states
        self.forward_batch = forward_batch
        self.qkv_latent_func = qkv_latent_func
        self.hidden_states_ = None
        self.qkv_latent_ = None
        # When True, hidden_states_local is already attn_tp-gathered upstream
        # (e.g. by the input-scattered attention input step for DSA). fetch_* must NOT gather again.
        self.is_pre_gathered = is_pre_gathered

    def tp_all_gather_hidden_states(self, hidden_states, forward_batch):
        total_tokens = forward_batch.input_ids.shape[0]
        output = hidden_states.new_empty((total_tokens, hidden_states.shape[-1]))
        get_parallel().tp_group.all_gather_into_tensor(output, hidden_states)
        return output

    def fetch_qkv_latent(self):
        if self.qkv_latent_ is not None:
            return self.qkv_latent_
        assert self.qkv_latent_func is not None
        self.qkv_latent_ = self.qkv_latent_func(
            self.hidden_states_local, self.forward_batch
        )
        if get_attn_tp_context().input_scattered and not self.is_pre_gathered:
            self.qkv_latent_ = self.tp_all_gather_hidden_states(
                self.qkv_latent_, self.forward_batch
            )
        return self.qkv_latent_

    def fetch_hidden_states(self):
        if self.hidden_states_ is not None:
            return self.hidden_states_
        self.hidden_states_ = self.hidden_states_local
        if get_attn_tp_context().input_scattered and not self.is_pre_gathered:
            self.hidden_states_ = self.tp_all_gather_hidden_states(
                self.hidden_states_, self.forward_batch
            )
        return self.hidden_states_


class AttnTpContext:
    def __init__(self):
        self.allow_input_scattered = False
        self.is_dsa = False

    def init_context(self, q_lora_rank, is_dsa, is_mhc=False):
        # Only MHC pre-gathers hidden states before DSA attention, so non-MHC DSA
        # cannot use scattered inputs.
        self.is_dsa = is_dsa
        self.allow_input_scattered = (
            get_parallel().enable_attn_tp_input_scattered
            and (_is_cuda or _is_npu)
            and q_lora_rank is not None
            and (is_mhc or not is_dsa)
            and get_parallel().tp_size > 1
            and not is_dp_attention_enabled()
            and get_moe_a2a_backend().is_none()
            and not enable_moe_dense_fully_dp()
            and get_spec().speculative_algorithm != "EAGLE3"
        )
        if get_parallel().enable_attn_tp_input_scattered:
            if not self.allow_input_scattered:
                logging.info(
                    "attn_tp_input_scattered is not enabled while other conditions are not met"
                )
            else:
                logging.info("attn_tp_input_scattered is enabled")

    def use_input_scattered(self, forward_batch: ForwardBatch):
        return (
            self.allow_input_scattered
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.input_ids is not None
            and not forward_batch.can_run_tbo
        )

    @property
    def input_scattered(self):
        return get_forward().attn_input_scattered

    def set_attn_inputs(self, attn_inputs: AttentionInputs):
        get_forward().set("attn_inputs", attn_inputs)

    def fetch_qkv_latent(self):
        attn_inputs = get_forward().attn_inputs
        assert attn_inputs is not None
        return attn_inputs.fetch_qkv_latent()

    def fetch_hidden_states(self):
        attn_inputs = get_forward().attn_inputs
        assert attn_inputs is not None
        return attn_inputs.fetch_hidden_states()

    def clear_attn_inputs(self) -> None:
        get_forward().set("attn_inputs", None)

    @contextmanager
    def maybe_input_scattered(self, forward_batch: ForwardBatch):
        flag = self.use_input_scattered(forward_batch)
        forward = get_forward()
        # scoped() also restores when the forward raises — the old in-place
        # swap leaked the flag on exceptions.
        with forward.scoped(attn_input_scattered=flag):
            try:
                yield
            finally:
                forward.set("attn_inputs", None)


ATTN_TP_CONTEXT = AttnTpContext()


def get_attn_tp_context():
    return ATTN_TP_CONTEXT


def _redistribute_from_attn_tp_shards(tensor: torch.Tensor) -> torch.Tensor:
    gathered = get_local_dp_buffer(
        get_parallel().attn_tp_group, hidden_size=tensor.shape[-1]
    )
    attn_tp_all_gather_into_tensor(gathered, tensor)
    return gathered


def _redistribute_to_attn_tp_shards(tensor: torch.Tensor) -> torch.Tensor:
    parallel = get_parallel()
    return tensor.tensor_split(parallel.attn_tp_size)[parallel.attn_tp_rank]
