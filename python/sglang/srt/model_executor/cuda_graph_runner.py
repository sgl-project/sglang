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

"""Run the model with cuda graph."""

import bisect
from contextlib import contextmanager
from typing import Callable, List

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels
from vllm.distributed.parallel_state import graph_capture
from vllm.model_executor.custom_op import CustomOp

from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.sampler import SampleOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    InputMetadata,
    update_flashinfer_indices,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import monkey_patch_vllm_all_gather


def _to_torch(model: torch.nn.Module, reverse: bool = False):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub._forward_method = sub.forward_cuda
                setattr(sub, "is_torch_compile", False)
            else:
                sub._forward_method = sub.forward_native
                setattr(sub, "is_torch_compile", True)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse)


@contextmanager
def patch_model(
    model: torch.nn.Module, enable_compile: bool, tp_group: "GroupCoordinator"
):
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model)
            monkey_patch_vllm_all_gather()
            backup_ca_comm = tp_group.ca_comm
            tp_group.ca_comm = None
            yield torch.compile(model.forward, mode="max-autotune-no-cudagraphs")
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True)
            monkey_patch_vllm_all_gather(reverse=True)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024


class CudaGraphRunner:
    def __init__(
        self,
        model_runner: "ModelRunner",
        max_batch_size_to_capture: int,
        use_torch_compile: bool,
        disable_padding: bool,
    ):
        self.model_runner = model_runner
        self.graphs = {}
        self.input_buffers = {}
        self.output_buffers = {}
        self.flashinfer_handlers = {}
        self.graph_memory_pool = None
        self.disable_padding = disable_padding

        # Common inputs
        self.max_bs = max_batch_size_to_capture
        self.input_ids = torch.zeros((self.max_bs,), dtype=torch.int32, device="cuda")
        self.req_pool_indices = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device="cuda"
        )
        self.seq_lens = torch.zeros((self.max_bs,), dtype=torch.int32, device="cuda")
        self.position_ids_offsets = torch.ones(
            (self.max_bs,), dtype=torch.int32, device="cuda"
        )
        self.out_cache_loc = torch.zeros(
            (self.max_bs,), dtype=torch.int32, device="cuda"
        )

        # FlashInfer inputs
        self.flashinfer_kv_indptr = torch.zeros(
            (self.max_bs + 1,), dtype=torch.int32, device="cuda"
        )
        self.flashinfer_kv_indices = torch.zeros(
            (self.max_bs * model_runner.model_config.context_len,),
            dtype=torch.int32,
            device="cuda",
        )
        self.flashinfer_kv_last_page_len = torch.ones(
            (self.max_bs,), dtype=torch.int32, device="cuda"
        )
        if model_runner.sliding_window_size is None:
            self.flashinfer_workspace_buffer = (
                self.model_runner.flashinfer_workspace_buffer
            )
        else:
            self.flashinfer_workspace_buffer = (
                self.model_runner.flashinfer_workspace_buffer
            )

            self.flashinfer_kv_indptr = [
                self.flashinfer_kv_indptr,
                self.flashinfer_kv_indptr.clone(),
            ]
            self.flashinfer_kv_indices = [
                self.flashinfer_kv_indices,
                self.flashinfer_kv_indices.clone(),
            ]

        # Sampling inputs
        vocab_size = model_runner.model_config.vocab_size
        self.sampling_info = SamplingBatchInfo.dummy_one(self.max_bs, vocab_size)

        self.compile_bs = [1, 2, 4, 8, 16, 24, 32] if use_torch_compile else []

        if use_torch_compile:
            set_torch_compile_config()

    def can_run(self, batch_size: int):
        if self.disable_padding:
            return batch_size in self.graphs
        else:
            return batch_size <= self.max_bs

    def capture(self, batch_size_list: List[int]):
        self.batch_size_list = batch_size_list
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            for bs in batch_size_list:
                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        input_buffers,
                        output_buffers,
                        flashinfer_handler,
                    ) = self.capture_one_batch_size(bs, forward)
                    self.graphs[bs] = graph
                    self.input_buffers[bs] = input_buffers
                    self.output_buffers[bs] = output_buffers
                    self.flashinfer_handlers[bs] = flashinfer_handler

    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        # Common inputs
        input_ids = self.input_ids[:bs]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        position_ids_offsets = self.position_ids_offsets[:bs]
        out_cache_loc = self.out_cache_loc[:bs]

        # FlashInfer inputs
        if not _grouped_size_compiled_for_decode_kernels(
            self.model_runner.model_config.num_attention_heads
            // self.model_runner.tp_size,
            self.model_runner.model_config.get_num_kv_heads(self.model_runner.tp_size),
        ):
            use_tensor_cores = True
        else:
            use_tensor_cores = False
        if self.model_runner.sliding_window_size is None:
            flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.flashinfer_workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=use_tensor_cores,
                paged_kv_indptr_buffer=self.flashinfer_kv_indptr[: bs + 1],
                paged_kv_indices_buffer=self.flashinfer_kv_indices,
                paged_kv_last_page_len_buffer=self.flashinfer_kv_last_page_len[:bs],
            )
        else:
            flashinfer_decode_wrapper = []
            for i in range(2):
                flashinfer_decode_wrapper.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.flashinfer_workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=use_tensor_cores,
                        paged_kv_indptr_buffer=self.flashinfer_kv_indptr[i][: bs + 1],
                        paged_kv_indices_buffer=self.flashinfer_kv_indices[i],
                        paged_kv_last_page_len_buffer=self.flashinfer_kv_last_page_len[
                            :bs
                        ],
                    )
                )
        update_flashinfer_indices(
            ForwardMode.DECODE,
            self.model_runner,
            req_pool_indices,
            seq_lens,
            None,
            flashinfer_decode_wrapper,
        )

        # Run and capture
        def run_once():
            input_metadata = InputMetadata(
                forward_mode=ForwardMode.DECODE,
                sampling_info=self.sampling_info[:bs],
                batch_size=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                out_cache_loc=out_cache_loc,
                return_logprob=False,
                top_logprobs_nums=0,
                positions=(seq_lens - 1 + position_ids_offsets).to(torch.int64),
                flashinfer_decode_wrapper=flashinfer_decode_wrapper,
            )

            return forward(input_ids, input_metadata.positions, input_metadata)

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            run_once()

            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        with torch.cuda.graph(graph, pool=self.graph_memory_pool, stream=stream):
            out = run_once()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        self.graph_memory_pool = graph.pool()
        return graph, None, out, flashinfer_decode_wrapper

    def replay(self, batch: ScheduleBatch):
        assert batch.out_cache_loc is not None
        raw_bs = len(batch.reqs)

        # Pad
        index = bisect.bisect_left(self.batch_size_list, raw_bs)
        bs = self.batch_size_list[index]
        if bs != raw_bs:
            self.seq_lens.zero_()
            self.position_ids_offsets.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_bs] = batch.input_ids
        self.req_pool_indices[:raw_bs] = batch.req_pool_indices
        self.seq_lens[:raw_bs] = batch.seq_lens
        self.position_ids_offsets[:raw_bs] = batch.position_ids_offsets
        self.out_cache_loc[:raw_bs] = batch.out_cache_loc

        # FlashInfer inputs
        update_flashinfer_indices(
            ForwardMode.DECODE,
            self.model_runner,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            None,
            self.flashinfer_handlers[bs],
        )

        # Sampling inputs
        self.sampling_info.inplace_assign(raw_bs, batch.sampling_info)

        # Replay
        torch.cuda.synchronize()
        self.graphs[bs].replay()
        torch.cuda.synchronize()
        sample_output, logits_output = self.output_buffers[bs]

        # Unpad
        if bs != raw_bs:
            logits_output = LogitsProcessorOutput(
                next_token_logits=logits_output.next_token_logits[:raw_bs],
                next_token_logprobs=None,
                normalized_prompt_logprobs=None,
                input_token_logprobs=None,
                input_top_logprobs=None,
                output_top_logprobs=None,
            )
            sample_output = SampleOutput(
                sample_output.success[:raw_bs],
                sample_output.probs[:raw_bs],
                sample_output.batch_next_token_ids[:raw_bs],
            )

        # Extract logprobs
        if batch.return_logprob:
            logits_output.next_token_logprobs = torch.nn.functional.log_softmax(
                logits_output.next_token_logits, dim=-1
            )
            return_top_logprob = any(x > 0 for x in batch.top_logprobs_nums)
            if return_top_logprob:
                logits_metadata = LogitsMetadata(
                    forward_mode=ForwardMode.DECODE,
                    top_logprobs_nums=batch.top_logprobs_nums,
                )
                logits_output.output_top_logprobs = LogitsProcessor.get_top_logprobs(
                    logits_output.next_token_logprobs, logits_metadata
                )[1]

        return sample_output, logits_output
