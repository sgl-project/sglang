from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Callable

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner, patch_model
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.hip_model_runner import HiPModelRunner


class HiPCudaGraphRunner(CudaGraphRunner):
    model_runner: "HiPModelRunner"

    def __init__(self, model_runner: "HiPModelRunner"):
        super().__init__(model_runner)

    def can_run(self, forward_batch: ForwardBatch):
        use_cached_mask = forward_batch.hip_use_cached_mask
        num_stage_cached = forward_batch.hip_metadata_cached_stage

        if self.enable_dp_attention:
            min_num_tokens, max_num_tokens = min(forward_batch.global_num_tokens), max(
                forward_batch.global_num_tokens
            )
            is_bs_supported = forward_batch.can_run_dp_cuda_graph and (
                (
                    min_num_tokens == max_num_tokens
                    and (max_num_tokens, use_cached_mask, num_stage_cached)
                    in self.graphs
                )
                if self.disable_padding
                else max_num_tokens <= self.max_bs
            )
        else:
            is_bs_supported = (
                (forward_batch.batch_size, use_cached_mask, num_stage_cached)
                in self.graphs
                if self.disable_padding
                else forward_batch.batch_size <= self.max_bs
            )

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )
        return is_bs_supported and is_encoder_lens_supported

    def capture(self):
        with graph_capture() as graph_capture_context:
            num_stages = len(self.model_runner.hip_attention_config.layers[0].stages)
            for layer_config in self.model_runner.hip_attention_config.layers:
                assert num_stages == len(layer_config.stages)
            cache_configs = [(True, None)]
            for i_stage in range(num_stages):
                cache_configs.append((False, i_stage))

            self.stream = graph_capture_context.stream
            capture_bs = (
                tqdm.tqdm(self.capture_bs)
                if get_tensor_model_parallel_rank() == 0
                else self.capture_bs
            )
            for bs in capture_bs:
                for use_cached_mask, num_cached_stages in cache_configs:
                    with patch_model(
                        self.model_runner.model,
                        bs in self.compile_bs,
                        bs,
                        self.model_runner.tp_group,
                    ) as forward:
                        (
                            graph,
                            output_buffers,
                        ) = self.capture_one_batch_size(
                            bs, forward, use_cached_mask, num_cached_stages
                        )
                        graph_handle = (bs, use_cached_mask, num_cached_stages)
                        self.graphs[graph_handle] = graph
                        self.output_buffers[graph_handle] = output_buffers
                    # Save gemlite cache after each capture
                    save_gemlite_cache()

    def capture_one_batch_size(
        self,
        bs: int,
        forward: Callable,
        hip_use_cached_mask: bool = False,
        hip_num_cached_stages: int = 0,
    ):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # Common inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        if self.is_encoder_decoder:
            encoder_lens = self.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = self.mrope_positions[:, :bs]

        if self.enable_dp_attention:
            global_num_tokens = [bs] * self.tp_size
            gathered_buffer = self.gathered_buffer[: bs * self.tp_size]
        else:
            global_num_tokens = None
            gathered_buffer = None

        spec_info = self.get_spec_info(num_tokens, positions)

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            hip_metadata_cache_pool=self.model_runner.hip_metadata_cache_pool,
            hip_use_cached_mask=hip_use_cached_mask,
            hip_metadata_cached_stage=hip_num_cached_stages,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            top_logprobs_nums=[0] * bs,
            positions=positions,
            global_num_tokens=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=(
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            ),
        )

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )

        # Run and capture
        def run_once():
            logits_output = forward(input_ids, forward_batch.positions, forward_batch)
            return logits_output.next_token_logits, logits_output.hidden_states

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
        return graph, out

    def replay(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        if self.enable_dp_attention:
            index = bisect.bisect_left(
                self.capture_bs, max(forward_batch.global_num_tokens)
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_bs].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_bs].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)

        if self.is_encoder_decoder:
            self.encoder_lens[:raw_bs].copy_(forward_batch.encoder_lens)
        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_bs].copy_(forward_batch.mrope_positions)

        if hasattr(forward_batch.spec_info, "hidden_states"):
            self.hidden_states[:raw_num_token] = forward_batch.spec_info.hidden_states

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens,
            forward_batch.seq_lens_sum + (bs - raw_bs),
            self.encoder_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )

        # Replay
        key = (
            bs,
            forward_batch.hip_use_cached_mask,
            forward_batch.hip_metadata_cached_stage,
        )
        self.graphs[key].replay()
        next_token_logits, hidden_states = self.output_buffers[key]

        # Extract logprobs
        logits_output = LogitsProcessorOutput(
            next_token_logits=next_token_logits[:raw_num_token],
            hidden_states=(
                hidden_states[:raw_num_token] if hidden_states is not None else None
            ),
        )
        return logits_output
