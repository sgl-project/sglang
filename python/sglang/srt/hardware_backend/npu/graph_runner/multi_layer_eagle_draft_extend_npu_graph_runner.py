# Copyright 2024-2025 SGLang Team
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
"""Run the multi-layer eagle draft extend model with npu graph."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner import model_capture_mode
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
    MultiLayerEagleDraftExtendInputBuffers,
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.speculative.multi_layer_eagle_worker import (
        MultiLayerEagleDraftWorker,
    )


class MultiLayerEagleDraftExtendNpuGraphRunner(
    MultiLayerEagleDraftExtendCudaGraphRunner
):
    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker, step: int):
        super().__init__(eagle_worker, step)

    def init_buffers_and_capture(
        self,
        cuda_graph_buffers,
        offset,
        next_cuda_graph_runner,
    ):
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() == 0:
            print(
                f"[draft_extend_npu] init_buffers_and_capture start, step={getattr(self, 'step', '?')}, max_bs={self.max_bs}, num_tokens_per_bs={self.num_tokens_per_bs}"
            )
        # Shared buffer setup logic from the parent class.
        # Overridden only to replace FullCudaGraphBackend with NPUCudaGraphBackend.
        self.next_cuda_graph_runner = next_cuda_graph_runner
        seq_lens_cpu = cuda_graph_buffers["seq_lens_cpu"]
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        with torch.device(self.model_runner.device):
            input_ids = cuda_graph_buffers["input_ids"][
                offset : offset + self.max_num_token
            ]
            out_cache_loc = cuda_graph_buffers["out_cache_loc"][
                offset : offset + self.max_num_token
            ]
            positions = cuda_graph_buffers["positions"][
                offset : offset + self.max_num_token
            ]

            seq_lens = cuda_graph_buffers["seq_lens"]
            req_pool_indices = cuda_graph_buffers["req_pool_indices"]
            num_correct_drafts = cuda_graph_buffers["num_correct_drafts"]
            num_accept_tokens = cuda_graph_buffers["num_accept_tokens"]

            extend_seq_lens = torch.full(
                (self.max_bs,),
                self.num_tokens_per_bs,
                dtype=torch.int32,
            )
            extend_start_loc = torch.arange(
                0,
                self.max_bs * self.num_tokens_per_bs,
                step=self.num_tokens_per_bs,
                dtype=torch.int32,
            )

            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)

            hidden_states = torch.zeros(
                (
                    self.max_num_token,
                    EagleDraftExtendInput.hidden_size_for(self.eagle_worker),
                ),
                dtype=EagleDraftExtendInput.dtype_for(self.eagle_worker),
            )

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

            if hasattr(self.model_runner.model_config.hf_config, "draft_vocab_size"):
                vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size
            elif hasattr(self.model_runner.model_config.hf_config, "hot_vocab_size"):
                vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size
            else:
                vocab_size = self.model_runner.model_config.vocab_size

            next_token_logits_buffer = torch.zeros(
                (
                    (
                        self.max_bs * self.num_tokens_per_bs
                        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2
                        else self.max_bs
                    ),
                    vocab_size,
                ),
                dtype=torch.float,
            )

        self.buffers = MultiLayerEagleDraftExtendInputBuffers(
            input_ids=input_ids,
            out_cache_loc=out_cache_loc,
            positions=positions,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=num_accept_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            mrope_positions=mrope_positions,
            hidden_states=hidden_states,
            next_token_logits_buffer=next_token_logits_buffer,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )

        self.backend = NPUCudaGraphBackend(
            self,
            enable_memory_saver=self.model_runner.server_args.enable_memory_saver,
        )

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() == 0:
            print(f"[draft_extend_npu] init_buffers_and_capture done")

    def replay(self, forward_batch: ForwardBatch, init_state: bool = True):
        # Same as parent replay() but uses replay_with_input_update
        # for NPU's seq_lens update before graph replay.
        assert forward_batch.out_cache_loc is not None
        self.deepep_adapter.replay()
        buffers = self.buffers

        raw_bs = forward_batch.batch_size
        num_tokens = raw_bs * self.num_tokens_per_bs
        if self.require_mlp_tp_gather:
            max_batch_size = max(forward_batch.original_global_num_tokens_cpu)
            bs = self._pad_to_bucket(int(max_batch_size), self.capture_bs)
        else:
            bs = self._pad_to_bucket(raw_bs, self.capture_bs)

        if init_state:
            self.init_replay_state(forward_batch, bs, raw_bs, num_tokens)

        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(bs * self.num_tokens_per_bs)

        forward_batch.spec_info.hidden_states = buffers.hidden_states[:num_tokens]
        forward_batch.spec_info.num_correct_drafts = buffers.num_correct_drafts[:bs]
        forward_batch.spec_info.num_accept_tokens = buffers.num_accept_tokens[:bs]
        forward_batch.spec_info.num_tokens_per_req = self.num_tokens_per_bs
        forward_batch.spec_info.num_tokens_for_logprob_per_req = 1
        forward_batch.spec_info.positions = buffers.positions[:num_tokens]
        forward_batch.spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[:bs]

        from types import SimpleNamespace

        fb_view = SimpleNamespace(
            batch_size=bs,
            forward_mode=self.forward_mode,
            input_ids=getattr(forward_batch, "input_ids", None),
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum
            + (bs - raw_bs) * self.seq_len_fill_value,
            seq_lens_cpu=buffers.seq_lens_cpu,
            encoder_lens=None,
            out_cache_loc=buffers.out_cache_loc[:num_tokens],
            spec_info=forward_batch.spec_info,
        )
        self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].init_forward_metadata_out_graph(fb_view)

        self.raw_bs = raw_bs
        self.bs = bs
        shape_key = self._make_graph_key(bs)

        # NPU: update seq_lens in captured graph before replay
        seq_lens = buffers.seq_lens_cpu[:raw_bs].tolist() + [0] * (bs - raw_bs)
        out = self.backend.replay_with_input_update(
            shape_key,
            seq_lens=seq_lens,
            attr_name="actual_seq_kvlen",
            attr_type=[],
        )

        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2:
            unpadding_bs = num_tokens
        elif bs != raw_bs:
            forward_batch.spec_info.num_correct_drafts = buffers.num_correct_drafts[
                :raw_bs
            ]
            forward_batch.spec_info.num_accept_tokens = buffers.num_accept_tokens[
                :raw_bs
            ]
            unpadding_bs = raw_bs
        else:
            unpadding_bs = None

        if unpadding_bs is not None:
            out_copy = out
            out = LogitsProcessorOutput(
                next_token_logits=out.next_token_logits[:unpadding_bs],
                hidden_states=out.hidden_states[:unpadding_bs],
            )
            out.topk_p = out_copy.topk_p[:raw_bs]
            out.topk_index = out_copy.topk_index[:raw_bs]
        return out

    def _create_graph(self):
        return torch.npu.NPUGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.npu.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.npu.graph(
            graph,
            pool=pool,
            stream=stream,
            auto_dispatch_capture=True,
        ):
            out = run_once_fn()
        return out


class MultiLayerEagleMultiStepDraftExtendNpuGraphRunner(
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
):
    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker):
        super().__init__(eagle_worker)

    def _init_and_capture(self):
        if cuda_graph_fully_disabled():
            self.runners = [None] * self.speculative_num_steps
            return

        self.runners: List[Optional[MultiLayerEagleDraftExtendNpuGraphRunner]] = []
        buffer_len_list: List[int] = []

        for step in range(self.speculative_num_steps):
            if self.draft_extend_attn_backend_list[step]:
                runner = MultiLayerEagleDraftExtendNpuGraphRunner(
                    self.eagle_worker, step
                )
                self.runners.append(runner)

                self.seq_len_fill_value = runner.seq_len_fill_value
                self.max_bs = runner.max_bs
                buffer_len_list.append(runner.max_num_token)
                self.offsets.append(self.offsets[-1] + runner.max_num_token)
            else:
                self.runners.append(None)

        self.cuda_graph_buffers["seq_lens_cpu"] = torch.full(
            (self.max_bs,),
            self.seq_len_fill_value,
            dtype=torch.int32,
        )

        with torch.device(self.device):
            self.cuda_graph_buffers["input_ids"] = torch.zeros(
                (self.offsets[-1],), dtype=torch.int64
            )
            self.cuda_graph_buffers["out_cache_loc"] = torch.ones(
                (self.offsets[-1],), dtype=torch.int64
            )
            self.cuda_graph_buffers["positions"] = torch.zeros(
                (self.offsets[-1],), dtype=torch.int64
            )

            self.cuda_graph_buffers["seq_lens"] = torch.full(
                (self.max_bs,),
                self.seq_len_fill_value,
                dtype=torch.int32,
            )
            self.cuda_graph_buffers["req_pool_indices"] = torch.zeros(
                (self.max_bs,), dtype=torch.int64
            )
            self.cuda_graph_buffers["num_correct_drafts"] = torch.full(
                (self.max_bs,), 1, dtype=torch.int32
            )
            self.cuda_graph_buffers["num_accept_tokens"] = torch.full(
                (self.max_bs,), 1, dtype=torch.int32
            )

        for step in range(self.speculative_num_steps - 1, -1, -1):
            if self.runners[step] is not None:
                tic = time.perf_counter()
                before_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph begin (step {step}). This can take up to several minutes. avail mem={before_mem:.2f} GB"
                )

                self.runners[step].init_buffers_and_capture(
                    self.cuda_graph_buffers,
                    self.offsets[step],
                    (
                        self.runners[step + 1]
                        if step + 1 < self.speculative_num_steps
                        else None
                    ),
                )

                after_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
                )
