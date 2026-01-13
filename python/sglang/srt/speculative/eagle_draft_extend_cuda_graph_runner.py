from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
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
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.spec_utils import fast_topk
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


class EAGLEDraftExtendCudaGraphRunner:
    def __init__(self, eagle_worker: EAGLEWorker):
        # Parse args
        self.eagle_worker = eagle_worker
        if not hasattr(eagle_worker, "model_runner"):
            # V2: EagleDraftWorker
            self.model_runner = model_runner = eagle_worker.draft_runner
            self.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        else:
            self.model_runner = model_runner = eagle_worker.model_runner
            self.forward_mode = ForwardMode.DRAFT_EXTEND

        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.dp_size
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.padded_static_len = -1

        # Attention backend
        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2:
            self.num_tokens_per_bs = self.speculative_num_draft_tokens
        else:
            self.num_tokens_per_bs = self.speculative_num_steps + 1
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        self.eagle_worker.draft_extend_attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        self.seq_len_fill_value = self.eagle_worker.draft_extend_attn_backend.get_cuda_graph_seq_len_fill_value()
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs

        # Separate buffer for V2 draft length - MUST NOT be modified by Phase 2
        # This ensures spec_info.accept_length always shows draft length (32),
        # not verify accept lengths, during CUDA graph replay.
        with torch.device(model_runner.device):
            self.draft_lens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Graph inputs
        with torch.device(model_runner.device):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.out_cache_loc = torch.ones(
                (self.max_num_token,), dtype=self._cache_loc_dtype()
            )
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros(
                (3, self.max_num_token), dtype=torch.int64
            )

            if (
                self.eagle_worker.speculative_algorithm.is_eagle3()
                and self.eagle_worker.eagle_use_aux_hidden_state
            ):
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
            self.seq_len_fill_value = (
                self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
            )
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.extend_seq_lens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            self.accept_length = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )

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
                vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size
            elif hasattr(
                self.model_runner.model_config.hf_config, "hot_vocab_size"
            ):  # llama_eagle3
                vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size
            else:
                vocab_size = self.model_runner.model_config.vocab_size

            self.next_token_logits_buffer = torch.zeros(
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
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max(forward_batch.global_num_tokens_cpu)
            )
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

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _cache_loc_dtype(self):
        return torch.int64

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self, forward_batch: ForwardBatch):
        self.graphs[self.bs].replay()

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(self, bs: int, forward: Callable, stream_idx: int = 0):
        graph = self._create_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        seq_lens_cpu = self.seq_lens_cpu[:bs]
        extend_seq_lens = self.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        mrope_positions = self.mrope_positions[:, :num_tokens]
        hidden_states = self.hidden_states[:num_tokens]
        accept_length = self.accept_length[:bs]
        next_token_logits_buffer = self.next_token_logits_buffer[
            : bs if self.forward_mode == ForwardMode.DRAFT_EXTEND else num_tokens
        ]

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

        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            accept_length=accept_length,
        )
        spec_info.positions = None

        self.deepep_adapter.capture(is_extend_in_batch=True)

        # Forward batch
        forward_batch = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
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
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            attn_backend=self.eagle_worker.draft_extend_attn_backend,
            padded_static_len=self.padded_static_len,
        )

        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=self.forward_mode,
            spec_info=spec_info,
        )

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            # Backup two fields, which will be modified in-place in `draft_forward`.
            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
            probs = torch.softmax(ret.next_token_logits, dim=-1)
            ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)

            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        self._capture_init(run_once)

        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def replay_prepare_metadata(self, forward_batch: ForwardBatch):
        """Phase 1: Metadata Preparation (Runs on Plan Stream).

        Handles buffer resizing, independent metadata copies (seq_lens, indices),
        and FlashInfer plan computation. DOES NOT access hidden_states, input_ids
        data, or accept_length to avoid race conditions with Main Stream.

        After calling this, replay_prepare_data() must be called on Main Stream,
        then replay() with skip_prepare=True.
        """
        assert forward_batch.out_cache_loc is not None

        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        raw_bs = forward_batch.batch_size
        num_tokens = forward_batch.input_ids.shape[0]
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]
        if bs * self.num_tokens_per_bs != num_tokens:
            # Reset buffers for padding
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.out_cache_loc.zero_()
            self.positions.zero_()
            self.accept_length.fill_(self.num_tokens_per_bs)
            self.extend_seq_lens.fill_(self.num_tokens_per_bs)

        # Copy INDEPENDENT metadata (safe to run on Plan Stream)
        # Note: We intentionally skip input_ids, hidden_states, accept_length here
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        if forward_batch.extend_seq_lens is not None:
            self.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens)
        else:
            self.extend_seq_lens[:raw_bs].fill_(self.num_tokens_per_bs)
        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        self.positions[:num_tokens].copy_(forward_batch.positions)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)

        # TODO(ch-wan): support num_token_non_padded
        if self.require_gathered_buffer:
            self.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            self.global_num_tokens_for_logprob_gpu.fill_(bs)

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(self.seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        if forward_batch.extend_seq_lens_cpu is not None:
            self.extend_seq_lens_cpu[:raw_bs] = forward_batch.extend_seq_lens_cpu
        else:
            self.extend_seq_lens_cpu[:raw_bs] = [self.num_tokens_per_bs] * raw_bs
        if bs > raw_bs:
            self.extend_seq_lens_cpu[raw_bs:bs] = [self.num_tokens_per_bs] * (
                bs - raw_bs
            )
        forward_batch.spec_info.extend_seq_lens_cpu = list(
            self.extend_seq_lens_cpu[:bs]
        )
        forward_batch.spec_info.extend_seq_lens_tensor = self.extend_seq_lens[:bs]

        if bs != raw_bs:
            forward_batch.spec_info.positions = self.positions[:num_tokens]
            # Use buffer accept_length for padding cases
            forward_batch.spec_info.accept_length = self.accept_length[:bs]
        elif self.forward_mode == ForwardMode.DRAFT_EXTEND_V2:
            # CRITICAL: Use separate draft_lens buffer, NOT accept_length!
            # Phase 2 modifies accept_length with verify results, but we need
            # spec_info.accept_length to always show draft length (32) during replay.
            forward_batch.spec_info.accept_length = self.draft_lens[:bs]

        # FlashInfer plan (the heavy lifting we want to overlap)
        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum
            + (bs - raw_bs) * self.seq_len_fill_value,
            encoder_lens=None,
            forward_mode=self.forward_mode,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu,
        )

        # Store state for replay_prepare_data() and replay()
        self.raw_bs = raw_bs
        self.bs = bs
        self.num_tokens = num_tokens

    def replay_prepare_data(
        self, forward_batch: ForwardBatch, accept_length: Optional[torch.Tensor] = None
    ):
        """Phase 2: Data Copy (Runs on Main Stream after wait_stream).

        Copies data that depends on Verify output: hidden_states, input_ids,
        and accept_length. Must be called after replay_prepare_metadata() and
        after Main Stream has synchronized with Plan Stream.

        Args:
            forward_batch: The forward batch.
            accept_length: The accept_length tensor from verify result. Required
                          for V2 mode to update the buffer with actual values.
        """
        raw_bs = self.raw_bs
        num_tokens = self.num_tokens

        # Copy DEPENDENT data (requires Verify to have completed)
        self.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        if (
            forward_batch.spec_info.hidden_states.shape[1]
            == self.hidden_states.shape[1]
        ):
            self.hidden_states[:num_tokens].copy_(forward_batch.spec_info.hidden_states)

        # Copy accept_length from verify result into buffer
        if accept_length is not None:
            self.accept_length[:raw_bs].copy_(accept_length)
        # Note: forward_batch.spec_info.accept_length already points to
        # self.accept_length[:bs] from Phase 1, so buffer updates are reflected

    def replay_prepare(self, forward_batch: ForwardBatch):
        """Prepare buffers and metadata for CUDA graph replay.

        This method can be called on a separate stream (e.g., Plan Stream) to
        overlap with other GPU work. After calling this, replay() should be
        called with skip_prepare=True.

        Note: This is kept for backward compatibility. For V2 with Plan Stream
        overlap, use replay_prepare_metadata() + replay_prepare_data() instead.
        """
        self.replay_prepare_metadata(forward_batch)
        self.replay_prepare_data(forward_batch)

    def replay(self, forward_batch: ForwardBatch, skip_prepare: bool = False):
        """Replay the captured CUDA graph.

        Args:
            forward_batch: The forward batch to replay.
            skip_prepare: If True, assumes replay_prepare() was already called
                         (e.g., on Plan Stream). If False, calls replay_prepare()
                         internally (original behavior).
        """
        self.deepep_adapter.replay()

        if not skip_prepare:
            self.replay_prepare(forward_batch)

        # Replay the graph
        self._replay(forward_batch)
        out = self.output_buffers[self.bs]

        if self.forward_mode == ForwardMode.DRAFT_EXTEND_V2:
            # DRAFT_EXTEND_V2: all tokens calculations whether accepted or not.
            unpadding_bs = self.num_tokens
        elif self.bs != self.raw_bs:
            forward_batch.spec_info.accept_length = self.accept_length[: self.raw_bs]
            unpadding_bs = self.raw_bs
        else:
            unpadding_bs = None

        if unpadding_bs is not None:
            out_copy = out
            out = LogitsProcessorOutput(
                next_token_logits=out.next_token_logits[:unpadding_bs],
                hidden_states=out.hidden_states[:unpadding_bs],
            )
            out.topk_p = out_copy.topk_p[:unpadding_bs]
            out.topk_index = out_copy.topk_index[:unpadding_bs]
        return out
