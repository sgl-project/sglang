from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Callable, List, Union


import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
)
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    LogitsProcessorOutput,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    enable_num_token_non_padded,
)
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


class EAGLEDecodeCudaGraphRunner:
    """
    Cuda graph runner for regular decoding step when speculative decoding is disabled
    dynamically. We merge the target model forward and draft forward into a single cuda graph.
    For draft model, we use draft_extend_attn_backend with decode mode.
    For target model, we use regular attn_backend with decode mode.
    """

    def __init__(self, eagle_worker: EAGLEWorker):
        # Parse args
        self.eagle_worker = eagle_worker
        self.model_runner = model_runner = eagle_worker.model_runner
        self.target_model_runner = eagle_worker.target_worker.model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)

        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.tp_size = self.model_runner.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.padded_static_len = -1
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.FULL
        self.num_tokens_per_bs = 1

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        if self.eagle_worker.target_decode_attn_backend is not None:
            self.eagle_worker.target_decode_attn_backend.init_cuda_graph_state(
                self.max_bs, self.max_num_token
            )
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )

        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Graph inputs
        with torch.device("cuda"):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.out_cache_loc = torch.zeros(
                (self.max_num_token,), dtype=self._cache_loc_dtype()
            )
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros(
                (3, self.max_num_token), dtype=torch.int64
            )

            if self.eagle_worker.speculative_algorithm.is_eagle3():
                self.hidden_states = torch.zeros(
                    (
                        self.max_num_token,
                        (
                            self.model_runner.model_config.hf_config.target_hidden_size
                            * 3
                            if hasattr(
                                self.model_runner.model_config.hf_config,
                                "target_hidden_size",
                            )
                            else self.model_runner.model_config.hidden_size * 3
                        ),
                    ),
                    dtype=self.model_runner.dtype,
                )
            else:
                self.hidden_states = torch.zeros(
                    (self.max_num_token, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                )

            self.seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            self.extend_seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int32)

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    self.global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    self.global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                self.global_num_tokens_gpu = None
                self.global_num_tokens_for_logprob_gpu = None

            if hasattr(
                self.model_runner.model_config.hf_config, "draft_vocab_size"
            ):  # llama_eagle
                draft_vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size
            elif hasattr(
                self.model_runner.model_config.hf_config, "hot_vocab_size"
            ):  # llama_eagle3
                draft_vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size
            else:
                draft_vocab_size = self.model_runner.model_config.vocab_size

            self.draft_next_token_logits_buffer = torch.zeros(
                (self.max_bs, draft_vocab_size),
                dtype=torch.float,
            )
            target_vocab_size = self.model_runner.model_config.vocab_size
            self.target_next_token_logits_buffer = torch.zeros(
                (self.max_bs, target_vocab_size),
                dtype=torch.float,
            )

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        if self.require_mlp_tp_gather:
            cuda_graph_bs = max(forward_batch.global_num_tokens_cpu)
        else:
            cuda_graph_bs = forward_batch.seq_lens.numel()

        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

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
        seq_lens_cpu = self.seq_lens_cpu[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        mrope_positions = self.mrope_positions[:, :num_tokens]
        hidden_states = self.hidden_states[:num_tokens]
        draft_next_token_logits_buffer = self.draft_next_token_logits_buffer[:bs]
        target_next_token_logits_buffer = self.target_next_token_logits_buffer[:bs]
        target_hidden_states = self.hidden_states[:bs]

        if self.require_mlp_tp_gather:
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=self.input_ids.device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [bs] * self.dp_size,
                    dtype=torch.int32,
                    device=self.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.input_ids.device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [bs],
                    dtype=torch.int32,
                    device=self.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        self.deepep_adapter.capture(is_extend_in_batch=False)

        # Forward batch for target
        target_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=target_next_token_logits_buffer,
            req_to_token_pool=self.target_model_runner.req_to_token_pool,
            token_to_kv_pool=self.target_model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=self.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=self.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            attn_backend=self.eagle_worker.target_decode_attn_backend,
            padded_static_len=self.padded_static_len,
        )

        draft_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=draft_next_token_logits_buffer,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=self.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=self.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            attn_backend=self.eagle_worker.draft_extend_attn_backend,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            spec_algorithm=self.model_runner.spec_algorithm,
            padded_static_len=self.padded_static_len,
        )
        draft_forward_batch.target_hidden_states = target_hidden_states
        if self.eagle_worker.target_decode_attn_backend is not None:
            self.eagle_worker.target_decode_attn_backend.init_forward_metadata_capture_cuda_graph(
                bs=bs,
                num_tokens=num_tokens,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=None,
            )

        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            target_forward_batch.dp_local_start_pos = None
            target_forward_batch.dp_local_num_tokens = None
            draft_forward_batch.dp_local_start_pos = None
            draft_forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                target_forward_batch.dp_padding_mode.is_max_len(),
            )
            kwargs = {}
            # TODO: Enable PP later
            # if (
            #     self.pp_size > 1
            #     and "pp_proxy_tensors" in inspect.signature(forward).parameters
            # ):
            #     kwargs["pp_proxy_tensors"] = PPProxyTensors(
            #         {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
            #     )
            # We are using a separate attn_backend to avoid changing target_model_runner
            # original attn_backend.
            target_ret = self.target_model_runner.forward_decode(
                target_forward_batch,
                skip_attn_backend_init=True,
                pp_proxy_tensors=None,
            )
            self.hidden_states[:num_tokens].copy_(target_ret.hidden_states)
            draft_ret =  self.model_runner.model.forward(
                draft_forward_batch.input_ids,
                draft_forward_batch.positions,
                draft_forward_batch,
                **kwargs,
            )

            # We intentionally skip topk_index and topk_p capturing and put it outside
            # of the cuda graph to be able to skip it unless needed.
            # TODO: evaluate if we should either: 1. always run topk and include in cuda graph
            # 2. only run when turning spec decode back on but has kernel launch overhead.
            return (target_ret, draft_ret)

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())
        out = CudaGraphRunner._capture_graph(
            self, graph, get_global_graph_memory_pool(), stream, run_once
        )
        return graph, out

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
    ):
        # TODO: Check if recapture is needed
        # self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = max_num_tokens
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)

        seq_lens_cpu = None
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(self.seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
            seq_lens_cpu = self.seq_lens_cpu[:bs]

        # if pp_proxy_tensors:
        #     for key in self.pp_proxy_tensors.keys():
        #         dim = pp_proxy_tensors[key].shape[0]
        #         self.pp_proxy_tensors[key][:dim].copy_(pp_proxy_tensors[key])

        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_num_token].copy_(forward_batch.mrope_positions)
        if self.require_gathered_buffer:
            self.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            self.global_num_tokens_for_logprob_gpu.fill_(bs * self.num_tokens_per_bs)
        if enable_num_token_non_padded(self.model_runner.server_args):
            num_token_non_padded = forward_batch.num_token_non_padded
            if self.require_gathered_buffer:
                tokens_per_rank = bs // self.attn_tp_size * self.num_tokens_per_bs
                num_local_token_non_padded = torch.clamp(
                    num_token_non_padded - tokens_per_rank * self.attn_tp_rank,
                    min=0,
                    max=tokens_per_rank,
                )
                self.num_token_non_padded.copy_(num_local_token_non_padded)
            else:
                self.num_token_non_padded.copy_(num_token_non_padded)
        if self.enable_two_batch_overlap:
            raise NotImplementedError("Two batch overlap is not supported yet")
        if forward_batch.forward_mode.is_idle() and forward_batch.spec_info is not None:
            forward_batch.spec_info.custom_mask = self.custom_mask
        # Target Attention Backend
        if self.eagle_worker.target_decode_attn_backend is not None:
            self.eagle_worker.target_decode_attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                self.req_pool_indices[:bs],
                self.seq_lens[:bs],
                forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
                None,
                self.capture_forward_mode,
                forward_batch.spec_info,
                seq_lens_cpu=seq_lens_cpu,
            )
        
        # Draft Attention backend
        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
            None,
            self.capture_forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=seq_lens_cpu,
        )

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
    ) -> List[Union[LogitsProcessorOutput, PPProxyTensors]]:
        self.deepep_adapter.replay()

        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch)
        else:
            # In speculative decoding, these two fields are still needed.
            self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.positions[: self.raw_num_token].copy_(forward_batch.positions)

        # Replay
        self.graphs[self.bs].replay()

        # TODO: Support PP
        output = self.output_buffers[self.bs]

        target_output, draft_output = output

        return LogitsProcessorOutput(
            next_token_logits=target_output.next_token_logits[: self.raw_num_token],
            hidden_states=(
                target_output.hidden_states[: self.raw_num_token]
                if target_output.hidden_states is not None
                else None
            ),
        ), LogitsProcessorOutput(
            next_token_logits=draft_output.next_token_logits[: self.raw_num_token],
            hidden_states=(
                draft_output.hidden_states[: self.raw_num_token]
                if draft_output.hidden_states is not None
                else None
            ),
        )

    def _cache_loc_dtype(self):
        return torch.int64