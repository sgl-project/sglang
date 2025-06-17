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

import bisect
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    LogitsProcessorOutput,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.eagle_utils import EagleDraftInput, create_draft_kv_indices
from sglang.srt.utils import fast_topk, is_cuda, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.speculative.naive_eagle import NaiveEagleWorker

if is_cuda():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob


class NaiveEAGLECudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(
        self,
        eagle_worker: NaiveEagleWorker,
    ):
        # Parse args
        self.model_runner = eagle_worker.target_worker.model_runner
        self.draft_model_runner = eagle_worker.model_runner
        self.requests_all_greedy = eagle_worker.requests_all_greedy
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = self.model_runner.server_args.enable_torch_compile
        self.disable_padding = self.model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = self.model_runner.model_config.is_encoder_decoder
        self.enable_dp_attention = self.model_runner.server_args.enable_dp_attention
        self.enable_sp_layernorm = self.model_runner.server_args.enable_sp_layernorm
        self.speculative_algorithm = self.model_runner.server_args.speculative_algorithm
        self.tp_size = self.model_runner.server_args.tp_size
        self.dp_size = self.model_runner.server_args.dp_size
        self.enable_profile_cuda_graph = (
            self.model_runner.server_args.enable_profile_cuda_graph
        )
        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(self.model_runner)
        self.capture_hidden_mode = CaptureHiddenMode.NULL

        self.num_tokens_per_bs = 2
        self.capture_forward_mode = ForwardMode.NAIVE_TARGET_VERIFY

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)
        self.draft_model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)
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
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)

            self.hidden_states = torch.empty(
                (self.max_num_token, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,
            )

            # NOTE:Add for naive eagle
            self.accept_index = torch.full((self.max_bs, 2), -1, dtype=torch.int32)
            self.accept_length = torch.zeros((self.max_bs,), dtype=torch.int32)

            # NOTE: Add for no greedy requests
            self.one_tensor = torch.tensor([1])
            self.spec_info_topk_p = torch.zeros((self.max_bs, 1), dtype=torch.float32)
            self.spec_info_topk_index = torch.zeros((self.max_bs, 1), dtype=torch.int64)
        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
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
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        mrope_positions = self.mrope_positions[:, :bs]

        spec_info_topk_p = self.spec_info_topk_p[:bs]
        spec_info_topk_index = self.spec_info_topk_index[:bs]

        verify_spec_info, draft_spec_info = self.get_spec_info()
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                verify_spec_info.capture_hidden_mode
                if verify_spec_info
                else CaptureHiddenMode.NULL
            )

        temperatures = torch.ones((bs, 1), dtype=torch.float32, device="cuda")
        top_ps = torch.ones((bs,), dtype=torch.float32, device="cuda")
        top_ks = torch.ones((bs,), dtype=torch.int32, device="cuda")
        sampling_info = SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=None,
            is_all_greedy=False,
            need_min_p_sampling=None,
            vocab_size=None,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
        )

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            encoder_lens=None,
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=verify_spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            sampling_info=sampling_info,
            naive_skip_attn_backend_init=True,
        )
        draft_token_num = 2
        kv_indptr = torch.zeros(
            size=[1 + draft_token_num * bs], dtype=torch.int32, device="cuda"
        )
        kv_indices = torch.zeros(
            size=[
                forward_batch.seq_lens_sum * draft_token_num
                + (draft_token_num + 1) * bs
            ],
            dtype=torch.int32,
            device="cuda",
        )
        forward_batch.spec_info.kv_indices = kv_indices
        forward_batch.spec_info.kv_indptr = kv_indptr

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            None,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )

        # Draft Attention backend
        accept_length = self.accept_length[:bs]
        draft_spec_info.accept_length = accept_length + 2

        forward_batch.forward_mode = ForwardMode.NAIVE_DRAFT_EXTEND
        forward_batch.spec_info = draft_spec_info
        self.draft_model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens + 2,
            None,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        forward_batch.forward_mode = self.capture_forward_mode
        forward_batch.spec_info = verify_spec_info

        # we add some infos for capture codes after `self.model_runner.forward`
        accept_index = self.accept_index[:bs]

        # Run and capture
        def run_once():
            logits_output, _ = self.model_runner.forward(
                forward_batch
            )  # target model verify
            # verify
            indices = torch.arange(bs, device="cuda", dtype=torch.int32)
            accept_index[:, 0] = indices * 2
            accept_index[:, 1] = -1  # init it
            if self.requests_all_greedy:

                probs = torch.softmax(logits_output.next_token_logits, dim=-1)
                _, token_indices = fast_topk(probs, topk=1, dim=-1)
                next_token_ids = token_indices.squeeze(-1)

                draft_token = input_ids[2 * indices + 1]
                target_token = next_token_ids[2 * indices]

                mask = draft_token == target_token
                accept_index[:, 1] = torch.where(mask, 2 * indices + 1, -1)
            else:
                # apply temperature and get target probs
                expanded_temperature = torch.repeat_interleave(
                    forward_batch.sampling_info.temperatures, 2, dim=0
                )  # (bs * draft_token_num, 1)

                target_probs = F.softmax(
                    logits_output.next_token_logits / expanded_temperature, dim=-1
                )  # (bs * draft_token_num, vocab_size)
                target_probs = top_k_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        forward_batch.sampling_info.top_ks, 2, dim=0
                    ),
                )  # (bs * draft_token_num, vocab_size)
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        forward_batch.sampling_info.top_ps, 2, dim=0
                    ),
                )
                target_verify_probs = target_probs[indices * 2]
                coins = torch.rand((bs), dtype=torch.float32, device="cuda")
                draft_p = spec_info_topk_p.squeeze()
                target_p = torch.gather(
                    target_verify_probs, dim=1, index=spec_info_topk_index
                ).squeeze(1)

                mask = coins < torch.min(self.one_tensor, target_p / draft_p)

                accept_index[:, 1] = torch.where(
                    mask, 2 * indices + 1, accept_index[:, 1]
                )

                # prepare next_token_ids
                next_token_ids = torch.multinomial(target_probs, num_samples=1).squeeze(
                    -1
                )

            accept_length.copy_((accept_index[:, 1] != -1).to(torch.int32))
            accept_length_for_draft_extend = torch.ones_like(
                accept_length, dtype=torch.int32, device="cuda"
            )

            draft_spec_info.hidden_states = logits_output.hidden_states
            draft_spec_info.accept_length = accept_length_for_draft_extend
            draft_spec_info.verified_id = next_token_ids
            draft_spec_info.seq_lens_for_draft_extend = forward_batch.seq_lens + (
                accept_length_for_draft_extend + 1
            )
            draft_spec_info.req_pool_indices_for_draft_extend = (
                forward_batch.req_pool_indices
            )
            forward_batch.spec_info = draft_spec_info
            draft_logits_output = self.forward_draft_extend_after_decode_cuda_graph(
                forward_batch, accept_index
            )

            return (
                logits_output.next_token_logits,
                logits_output.hidden_states,
                next_token_ids,
                accept_index,
                draft_logits_output.hidden_states,
                draft_logits_output.next_token_logits,
                draft_spec_info,
            )

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            forward_batch.forward_mode = self.capture_forward_mode
            verify_spec_info_backup = forward_batch.spec_info
            seq_lens_back_up = forward_batch.seq_lens
            positions_backup = forward_batch.positions
            input_ids_backup = forward_batch.input_ids
            run_once()
            forward_batch.spec_info = verify_spec_info_backup
            forward_batch.positions = positions_backup
            forward_batch.input_ids = input_ids_backup
            forward_batch.seq_lens = seq_lens_back_up
            forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool
            forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
            forward_batch.attn_backend = self.model_runner.attn_backend
            forward_batch.forward_mode = self.capture_forward_mode

        with torch.cuda.graph(
            graph, pool=get_global_graph_memory_pool(), stream=stream
        ):
            out = run_once()

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def replay_prepare(self, forward_batch: ForwardBatch):
        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        if self.enable_dp_attention or self.enable_sp_layernorm:
            index = bisect.bisect_left(
                self.capture_bs, sum(forward_batch.global_num_tokens_cpu)
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)

        self.spec_info_topk_index[:raw_bs].copy_(forward_batch.spec_info_topk_index)
        self.spec_info_topk_p[:raw_bs].copy_(forward_batch.spec_info_topk_p)

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(1)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        if self.is_encoder_decoder:
            self.encoder_lens[:raw_bs].copy_(forward_batch.encoder_lens)
        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_bs].copy_(forward_batch.mrope_positions)
        if self.enable_dp_attention or self.enable_sp_layernorm:
            self.global_num_tokens_gpu.copy_(forward_batch.global_num_tokens_gpu)

        if hasattr(forward_batch.spec_info, "hidden_states"):
            self.hidden_states[:raw_num_token] = forward_batch.spec_info.hidden_states

        forward_batch.forward_mode = self.capture_forward_mode

        seq_lens_sum = forward_batch.seq_lens_sum + (bs - raw_bs)
        seq_lens = self.seq_lens + 2
        draft_token_num = 2
        kv_indptr = torch.empty(
            size=[1 + draft_token_num * bs], dtype=torch.int32, device="cuda"
        )
        kv_indices = torch.empty(
            size=[seq_lens_sum * draft_token_num + (draft_token_num + 1) * bs],
            dtype=torch.int32,
            device="cuda",
        )
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        create_draft_kv_indices[(bs,)](
            kv_indptr,
            kv_indices,
            self.req_pool_indices,
            req_to_token,
            seq_lens,
            draft_token_num,
            req_to_token.shape[-1],
            next_power_of_2(bs),
        )
        forward_batch.spec_info.kv_indptr = kv_indptr
        forward_batch.spec_info.kv_indices = kv_indices
        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens,
            seq_lens_sum,
            None,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu,
        )

        self.verify_input = forward_batch.spec_info
        draft_input = EagleDraftInput()
        accept_length_for_draft_extend = (
            torch.ones((bs,), dtype=torch.int32, device="cuda") + 1
        )  # always 2 tokens
        draft_input.accept_length = accept_length_for_draft_extend
        # Draft Attention backend
        forward_batch.forward_mode = ForwardMode.NAIVE_DRAFT_EXTEND
        forward_batch.spec_info = draft_input

        self.draft_model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens + 2,
            (forward_batch.seq_lens_sum + 2 * bs) + (bs - raw_bs) * 2,
            None,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu,
        )
        forward_batch.forward_mode = self.capture_forward_mode
        forward_batch.spec_info = self.verify_input

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

    def replay(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch)
        else:
            # In speculative decoding, these two fields are still needed.
            self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.positions[: self.raw_num_token].copy_(forward_batch.positions)

        # Replay
        self.graphs[self.bs].replay()
        (
            next_token_logits,
            hidden_states,
            next_token_ids,
            accept_index,
            draft_hidden_states,
            draft_next_token_logits,
            draft_input,
        ) = self.output_buffers[self.bs]

        logits_output = LogitsProcessorOutput(
            next_token_logits=next_token_logits[: self.raw_num_token],
            hidden_states=(
                hidden_states[: self.raw_num_token]
                if hidden_states is not None
                else None
            ),
        )

        next_token_ids = next_token_ids[: self.raw_num_token]
        accept_index = accept_index[: self.raw_bs]
        draft_logits_output = LogitsProcessorOutput(
            next_token_logits=draft_next_token_logits[: self.raw_bs],
            hidden_states=draft_hidden_states[: self.raw_bs],
        )

        return (
            logits_output,
            next_token_ids,
            accept_index,
            draft_logits_output,
            draft_input,
        )

    def get_spec_info(self):
        verify_spec_info = None
        draft_spec_info = None
        if self.model_runner.spec_algorithm.is_naive_eagle():
            from sglang.srt.speculative.eagle_utils import (
                EagleDraftInput,
                EagleVerifyInput,
            )

            verify_spec_info = EagleVerifyInput(
                draft_token=None,
                custom_mask=None,
                positions=None,
                retrive_index=None,
                retrive_next_token=None,
                retrive_next_sibling=None,
                retrive_cum_len=None,
                draft_token_num=2,
                spec_steps=1,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
                topk=1,
            )
            draft_spec_info = EagleDraftInput(
                topk_p=None,
                topk_index=None,
                hidden_states=None,
                accept_length=None,
                accept_length_cpu=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )

        return verify_spec_info, draft_spec_info

    def forward_draft_extend_after_decode_cuda_graph(
        self, forward_batch: ForwardBatch, accept_index
    ):
        # Prepare metadata
        forward_batch.forward_mode = ForwardMode.NAIVE_DRAFT_EXTEND
        forward_batch.spec_info.prepare_extend_after_decode_for_naive_eagle(
            forward_batch,
            1,
        )
        forward_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch.return_logprob = False
        forward_batch.req_to_token_pool = self.draft_model_runner.req_to_token_pool
        forward_batch.token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        forward_batch.attn_backend = self.draft_model_runner.attn_backend
        forward_batch.positions = forward_batch.spec_info.positions
        # Run
        logits_output, _ = self.draft_model_runner.forward(
            forward_batch, skip_attn_backend_init=True
        )

        last = accept_index[:, 1]
        first = accept_index[:, 0]
        save_index = torch.where(last != -1, last, first)
        logits_output.hidden_states = logits_output.hidden_states[save_index]
        logits_output.next_token_logits = logits_output.next_token_logits[save_index]
        return logits_output
