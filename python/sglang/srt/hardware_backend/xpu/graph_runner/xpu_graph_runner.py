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
"""Run the model with xpu graph and torch.compile."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.utils import register_xpu_device_properties_for_dynamo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


_fake_ops_registered = False


def register_fake_ops():
    """Register fake/abstract implementations for XPU sgl_kernel ops so that
    torch.compile (Dynamo) can trace through them using FakeTensors for shape
    and dtype propagation, without executing the real GPU kernels.
    """
    global _fake_ops_registered
    if _fake_ops_registered:
        return
    _fake_ops_registered = True

    @torch.library.register_fake("sgl_kernel::fwd")
    def _(
        q,
        k,
        v,
        q_v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        sinks,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        is_rotary_interleaved,
        scheduler_metadata,
        num_kv_splits,
        pack_gqa,
        sm_margin,
        out=None,
    ):
        total_q = q.shape[0]
        num_heads_q = q.shape[1]
        head_size_v = v.shape[-1]
        if out is None:
            out = q.new_empty(total_q, num_heads_q, head_size_v)
        softmax_lse = q.new_empty(num_heads_q, total_q, dtype=torch.float32)
        # out_accum and softmax_lse_accum are intermediate split-kv buffers;
        # they are only read when num_kv_splits > 1, which is determined at
        # runtime.  Return empty tensors with correct rank so downstream ops
        # that index into the list do not fail shape propagation.
        out_accum = q.new_empty(0)
        softmax_lse_accum = q.new_empty(0, dtype=torch.float32)
        return (out, softmax_lse, out_accum, softmax_lse_accum)

    @torch.library.register_fake("sgl_kernel::flash_mla_decode")
    def _(
        out,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        seq_lens,
        page_table,
        workspace,
        sm_scale,
        num_kv_splits,
    ):
        return


class XPUGraphRunner(DecodeCudaGraphRunner):
    """A XPUGraphRunner runs the forward pass of a model with xpu graph and torch.compile."""

    @staticmethod
    def _apply_xpu_compile_config() -> None:
        """Apply XPU-specific torch.compile / dynamo settings.

        Called unconditionally before super().__init__() so that the settings
        are in place regardless of whether --enable-torch-compile is passed.
        The critical flag is suppress_errors: when the Intel IGC compiler
        crashes with SIGFPE on certain reduction kernels (ocloc -device bmg
        returns exit code 245), dynamo falls back to eager for that subgraph
        instead of propagating the crash.
        """
        import torch._dynamo.config

        torch._dynamo.config.suppress_errors = True

    def __init__(self, model_runner: ModelRunner):
        assert (
            not model_runner.server_args.enable_memory_saver
        ), "XPUGraphRunner does not support Torch Memory Saver yet."
        register_fake_ops()
        self._apply_xpu_compile_config()
        register_xpu_device_properties_for_dynamo()
        super().__init__(model_runner)

        assert (
            not self.enable_two_batch_overlap
        ), "XPUGraphRunner does not support two batch overlap yet."
        assert (
            not self.require_mlp_tp_gather
        ), "XPUGraphRunner does not support MLP TP gather yet."
        assert (
            not self.require_mlp_sync
        ), "XPUGraphRunner does not support MLP sync yet."
        assert (
            not self.require_gathered_buffer
        ), "XPUGraphRunner does not support gathered buffer yet."

    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        )
        torch.xpu.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.xpu.memory._dump_snapshot("xpu_graph_runner_memory_usage.pickle")
        torch.xpu.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by XPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_xpu_time_total"
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total"
            )
            + "\n\nMemory Usage is saved to xpu_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)
