from __future__ import annotations

import bisect
import time
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    LogitsProcessorOutput,
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


class EAGLEDraftExtendCudaGraphRunner:
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
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.is_hopper_backend = model_runner.server_args.attention_backend == "hopper"
        if self.is_hopper_backend:
            self.padded_static_len = -1
        else:
            self.padded_static_len = self.speculative_num_steps + 1

        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)

        # Attention backend
        self.num_tokens_per_bs = self.speculative_num_steps + 1
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        self.eagle_worker.draft_extend_attn_backend.init_cuda_graph_state(
            self.max_num_token
        )
        self.seq_len_fill_value = (
            self.eagle_worker.draft_extend_attn_backend.get_cuda_graph_seq_len_fill_value()
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Graph inputs
        with torch.device("cuda"):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.out_cache_loc = torch.ones((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.hidden_states = torch.zeros(
                (self.max_num_token, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,
            )
            if self.is_hopper_backend:
                self.seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
                self.extend_seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
                self.accept_length = torch.ones((self.max_bs,), dtype=torch.int32)
            else:
                self.seq_lens = torch.full(
                    (self.max_bs,), self.padded_static_len, dtype=torch.int32
                )
                self.extend_seq_lens = (
                    torch.ones((self.max_bs,), dtype=torch.int32)
                    * self.num_tokens_per_bs
                )
                self.accept_length = (
                    torch.ones((self.max_bs,), dtype=torch.int32)
                    * self.num_tokens_per_bs
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
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )

    def can_run(self, forward_batch: ForwardBatch):
        batch_size = forward_batch.seq_lens.numel()

        is_bs_supported = (
            batch_size in self.graphs
            if self.disable_padding
            else batch_size <= self.max_bs
        )

        return is_bs_supported

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        extend_seq_lens = self.extend_seq_lens[:bs]
        accept_length = self.accept_length[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        hidden_states = self.hidden_states[:num_tokens]

        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            accept_length=accept_length,
            keep_indices=list(range(bs)),
        )
        spec_info.positions = None

        # Forward batch
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND,
            batch_size=bs,
            input_ids=input_ids,
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
            capture_hidden_mode=CaptureHiddenMode.LAST,
            attn_backend=self.eagle_worker.draft_extend_attn_backend,
            extend_seq_lens=extend_seq_lens,
            padded_static_len=self.padded_static_len,
        )

        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DRAFT_EXTEND,
            spec_info=spec_info,
        )

        # Run and capture
        def run_once():
            # Backup two fileds, which will be modified in-place in `draft_forward`.
            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.eagle_worker.draft_model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

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
        if self.is_hopper_backend:
            return self.replay_hopper(forward_batch)
        else:
            return self.replay_flashinfer(forward_batch)

    def replay_flashinfer(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        raw_bs = forward_batch.seq_lens.numel()
        num_tokens = forward_batch.input_ids.numel()
        assert raw_bs * self.num_tokens_per_bs == num_tokens

        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(self.padded_static_len)
            self.extend_seq_lens.fill_(1)
            self.accept_length.fill_(self.padded_static_len)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens[:raw_bs])
        self.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens[:raw_bs])
        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        self.positions[:num_tokens].copy_(forward_batch.positions)
        self.hidden_states[:num_tokens].copy_(forward_batch.spec_info.hidden_states)
        self.accept_length[:raw_bs].copy_(
            forward_batch.spec_info.accept_length[:raw_bs]
        )
        num_req = len(forward_batch.req_pool_indices)  # can be different from raw_bs
        self.req_pool_indices[:num_req].copy_(forward_batch.req_pool_indices)

        forward_batch.spec_info.accept_length = forward_batch.spec_info.accept_length[
            :bs
        ]

        forward_batch.spec_info.positions = None
        if bs != raw_bs:
            forward_batch.spec_info.keep_indices.extend(range(raw_bs, bs))
            if forward_batch.spec_info.all_padding_lens is not None:
                forward_batch.spec_info.all_padding_lens.extend(
                    [self.padded_static_len - 1] * (bs - raw_bs)
                )
            forward_batch.spec_info.accept_length = self.accept_length[:bs]

        assert len(forward_batch.spec_info.keep_indices) == bs
        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            seq_lens_sum=sum(forward_batch.seq_lens) + (bs - raw_bs),
            encoder_lens=None,
            forward_mode=ForwardMode.DRAFT_EXTEND,
            spec_info=forward_batch.spec_info,
        )

        # Replay
        self.graphs[bs].replay()
        out = self.output_buffers[bs]
        if bs != raw_bs:
            forward_batch.spec_info.accept_length = self.accept_length[:raw_bs]
            out = LogitsProcessorOutput(
                next_token_logits=out.next_token_logits[:raw_bs],
                hidden_states=out.hidden_states[:raw_bs],
            )
        return out

    def replay_hopper(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        raw_bs = forward_batch.seq_lens.numel()
        num_tokens = forward_batch.input_ids.numel()
        assert raw_bs * self.num_tokens_per_bs == num_tokens

        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.accept_length.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens[:raw_bs])
        self.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens[:raw_bs])
        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        self.positions[:num_tokens].copy_(forward_batch.positions)
        self.hidden_states[:num_tokens].copy_(forward_batch.spec_info.hidden_states)
        self.accept_length[:raw_bs].copy_(
            forward_batch.spec_info.accept_length[:raw_bs]
        )
        num_req = len(forward_batch.req_pool_indices)  # can be different from raw_bs
        self.req_pool_indices[:num_req].copy_(forward_batch.req_pool_indices)

        forward_batch.spec_info.positions = None
        if bs != raw_bs:
            forward_batch.spec_info.keep_indices.extend(range(raw_bs, bs))
            forward_batch.spec_info.accept_length = self.accept_length[:bs]

        assert len(forward_batch.spec_info.keep_indices) == bs
        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            seq_lens_sum=sum(forward_batch.seq_lens),
            encoder_lens=None,
            forward_mode=ForwardMode.DRAFT_EXTEND,
            spec_info=forward_batch.spec_info,
        )

        # Replay
        self.graphs[bs].replay()
        out = self.output_buffers[bs]
        if bs != raw_bs:
            forward_batch.spec_info.accept_length = self.accept_length[:raw_bs]
            out = LogitsProcessorOutput(
                next_token_logits=out.next_token_logits[:raw_bs],
                hidden_states=out.hidden_states[:raw_bs],
            )
        return out
