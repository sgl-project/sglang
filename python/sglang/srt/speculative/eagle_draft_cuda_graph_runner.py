from __future__ import annotations

import bisect
import time
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    set_global_graph_memory_pool,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_utils import EagleDraftInput

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


class EAGLEDraftCudaGraphRunner:
    def __init__(self, eagle_worker: EAGLEWorker):
        # Parse args
        self.eagle_worker = eagle_worker
        self.model_runner = model_runner = eagle_worker.model_runner
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.tp_size = self.model_runner.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        server_args = model_runner.server_args

        assert self.disable_padding

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.num_tokens_per_bs = server_args.speculative_eagle_topk

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.draft_attn_backend.init_cuda_graph_state(self.max_num_token)
        self.seq_len_fill_value = self.model_runner.draft_attn_backend.attn_backends[
            0
        ].get_cuda_graph_seq_len_fill_value()

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Graph inputs
        with torch.device("cuda"):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.out_cache_loc = torch.zeros(
                (self.max_num_token * self.speculative_num_steps,), dtype=torch.int64
            )
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.topk_p = torch.zeros((self.max_bs, self.topk), dtype=torch.float32)
            self.topk_index = torch.zeros((self.max_bs, self.topk), dtype=torch.int64)
            self.hidden_states = torch.zeros(
                (self.max_bs, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,
            )

        # Capture
        try:
            self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n"
                "Possible solutions:\n"
                "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "4. specify --dtype to the same dtype (e.g. bfloat16)\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )

    def can_run(self, forward_batch: ForwardBatch):
        is_bs_supported = (
            forward_batch.batch_size in self.graphs
            if self.disable_padding
            else forward_batch.batch_size <= self.max_bs
        )
        return is_bs_supported

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(self, num_seqs: int, forward: Callable):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
        num_tokens = num_seqs * self.num_tokens_per_bs

        # Graph inputs
        req_pool_indices = self.req_pool_indices[:num_seqs]
        seq_lens = self.seq_lens[:num_seqs]
        out_cache_loc = self.out_cache_loc[: num_tokens * self.speculative_num_steps]
        positions = self.positions[:num_tokens]
        topk_p = self.topk_p[:num_seqs]
        topk_index = self.topk_index[:num_seqs]
        hidden_states = self.hidden_states[:num_seqs]

        spec_info = EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
        )

        # Forward batch
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=num_seqs,
            input_ids=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            return_logprob=False,
            positions=positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=(
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            ),
        )

        # Attention backend
        self.model_runner.draft_attn_backend.init_forward_metadata_capture_cuda_graph(
            forward_batch
        )

        # Run and capture
        def run_once():
            # Backup two fileds, which will be modified in-place in `draft_forward`.
            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.eagle_worker.draft_forward(forward_batch)

            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            run_once()

            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        with torch.cuda.graph(
            graph, pool=get_global_graph_memory_pool(), stream=stream
        ):
            out = run_once()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def replay(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[: raw_num_token * self.speculative_num_steps].copy_(
            forward_batch.out_cache_loc
        )
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        self.topk_p[:raw_bs].copy_(forward_batch.spec_info.topk_p)
        self.topk_index[:raw_bs].copy_(forward_batch.spec_info.topk_index)
        self.hidden_states[:raw_bs].copy_(forward_batch.spec_info.hidden_states)

        # Attention backend
        self.model_runner.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
            forward_batch
        )

        # Replay
        self.graphs[bs].replay()

        return self.output_buffers[bs]
