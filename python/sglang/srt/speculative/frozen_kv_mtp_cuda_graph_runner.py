from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    ShapeKey,
    get_batch_sizes_to_capture,
    model_capture_mode,
)
from sglang.srt.model_executor.runner_backend.utils import resolve_decode_backend
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftInput
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPDraftWorker


@dataclass
class FrozenKVMTPInputBuffers(ForwardInputBuffers):
    req_pool_indices: torch.Tensor
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    hidden_states: torch.Tensor
    # Consumed by the captured seed iter; see `FrozenKVMTPDraftWorker.draft_forward`.
    bonus_tokens: torch.Tensor
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]


class FrozenKVMTPCudaGraphRunner(DecodeCudaGraphRunner):
    """CUDA graph runner for the Frozen-KV MTP recurrent draft-loop step.

    Subclasses DecodeCudaGraphRunner to inherit the outer capture loop
    (capture() / _capture_one_stream()), the bucket-padding helper
    (_pad_to_bucket), and the backend-driven capture/replay scaffolding.
    Frozen-KV-MTP-specific bits — the buffer dataclass, the dummy
    ForwardBatch + FrozenKVMTPDraftInput built in capture_one_shape, the
    target-KV-pool swap during capture, the worker's frozen-KV metadata
    helpers, the topk*topk bucket math, the expanded-bs bookkeeping, and
    the 3-tuple replay output — are overridden.

    Like the EAGLE draft runner, it does NOT call
    DecodeCudaGraphRunner.__init__ (that init sets up decode-only state);
    it sets up its own fields directly while satisfying the parent's
    capture() / backend contract.
    """

    def __init__(self, frozen_kv_mtp_worker: FrozenKVMTPDraftWorker):
        self.frozen_kv_mtp_worker = frozen_kv_mtp_worker
        self.model_runner = model_runner = frozen_kv_mtp_worker.draft_model_runner

        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.draft_attn_backend = frozen_kv_mtp_worker.draft_attn_backend
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )

        self.attn_backend = self.draft_attn_backend

        self.compile_bs = []
        self.enable_pdmux = False
        self.record_nolora_graph = False
        self.is_dllm = False

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.LAST

        self.num_tokens_per_bs = self.topk
        self.capture_bs, _ = get_batch_sizes_to_capture(
            model_runner, self.num_tokens_per_bs
        )
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        self.draft_attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)
        self.seq_len_fill_value = (
            self.draft_attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        seq_lens_cpu = torch.full(
            (self.max_num_token,), self.seq_len_fill_value, dtype=torch.int64
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        with torch.device(model_runner.device):
            req_pool_indices = torch.zeros((self.max_num_token,), dtype=torch.int64)
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)
            seq_lens = torch.full(
                (self.max_num_token,), self.seq_len_fill_value, dtype=torch.int64
            )
            topk_p = torch.zeros((self.max_bs, self.topk), dtype=torch.float32)
            topk_index = torch.zeros((self.max_bs, self.topk), dtype=torch.int64)
            hidden_states = torch.zeros(
                (self.max_bs, frozen_kv_mtp_worker._recurrent_hidden_size),
                dtype=self.model_runner.dtype,
            )
            bonus_tokens = torch.zeros((self.max_bs,), dtype=torch.int64)

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

        self.buffers = FrozenKVMTPInputBuffers(
            req_pool_indices=req_pool_indices,
            positions=positions,
            mrope_positions=mrope_positions,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            bonus_tokens=bonus_tokens,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )
        self.buffers.share_buffers()

        self.backend = resolve_decode_backend(self)

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture frozen-KV MTP cuda graph failed: {e}\n"
                f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _make_graph_key(self, bs, stream_idx=None, variant_label=None):
        return ShapeKey(size=bs)

    def _replay_graph(self, shape_key, forward_batch):
        return self.backend.replay(shape_key, forward_batch)

    def can_run_graph(self, forward_batch: ForwardBatch):
        if self.require_mlp_tp_gather:
            cuda_graph_bs = max(forward_batch.global_num_tokens_cpu) // (
                self.topk * self.topk
            )
        else:
            cuda_graph_bs = (
                forward_batch.batch_size // self.topk
                if self.topk > 1
                else forward_batch.batch_size
            )

        is_bs_supported = (
            self.backend.can_run(forward_batch, self._make_graph_key(cuda_graph_bs))
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )
        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph
        return is_bs_supported

    def capture_one_shape(
        self,
        size: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
    ):
        del forward, stream_idx, variant_label
        buffers = self.buffers
        request_bs = size
        expanded_bs = request_bs * self.num_tokens_per_bs

        req_pool_indices = buffers.req_pool_indices[:expanded_bs]
        positions = buffers.positions[:expanded_bs]
        mrope_positions = buffers.mrope_positions[:, :expanded_bs]
        seq_lens = buffers.seq_lens[:expanded_bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:expanded_bs]
        topk_p = buffers.topk_p[:request_bs]
        topk_index = buffers.topk_index[:request_bs]
        hidden_states = buffers.hidden_states[:request_bs]
        bonus_tokens = buffers.bonus_tokens[:request_bs]

        if self.require_mlp_tp_gather:
            global_num_tokens_cpu = [expanded_bs] * self.dp_size
        elif self.require_attn_tp_gather:
            global_num_tokens_cpu = [expanded_bs]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu,
                dtype=torch.int32,
                device=buffers.positions.device,
            )
            buffers.global_num_tokens_gpu.copy_(num_tokens_tensor)
            buffers.global_num_tokens_for_logprob_gpu.copy_(num_tokens_tensor)
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
        else:
            global_dp_buffer_len = None
            global_num_tokens = None
            global_num_tokens_for_logprob = None

        spec_info = FrozenKVMTPDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            bonus_tokens=bonus_tokens,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        spec_info.positions = positions

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=expanded_bs,
            input_ids=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=None,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                expanded_bs,
                forward_batch.dp_padding_mode.is_max_len(),
                global_num_tokens_cpu,
            )
            set_is_extend_in_batch(False)

            hidden_states_backup = forward_batch.spec_info.hidden_states
            # The capture batch is marked by the capture metadata helper
            # below, so draft_forward skips its eager plan.
            ret = self.frozen_kv_mtp_worker.draft_forward(forward_batch)
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        # Swap the draft backend's token_to_kv_pool to the frozen target pool
        # for the capture; the single backend-attr swap is seen by both
        # get_token_to_kv_pool() (via get_attn_backend()) and the
        # backend's own reads.
        target_pool = self.frozen_kv_mtp_worker.kv_context.target_token_to_kv_pool
        saved_backend_pool = self.draft_attn_backend.token_to_kv_pool
        self.draft_attn_backend.token_to_kv_pool = target_pool
        try:
            with forward_context(ForwardContext(attn_backend=self.draft_attn_backend)):
                self.frozen_kv_mtp_worker._init_frozen_kv_metadata_capture_cuda_graph(
                    forward_batch
                )
                self.deepep_adapter.capture(is_extend_in_batch=False)
                shape_key = self._make_graph_key(request_bs)
                post_warmup_hook = getattr(
                    self.draft_attn_backend, "on_after_cuda_graph_warmup", None
                )
                self._maybe_flashinfer_autotune_speculative_draft(
                    run_once,
                    post_warmup_hook=post_warmup_hook,
                    skip_logits=False,
                )
                self.backend.capture_one(
                    shape_key,
                    run_once,
                    dummies=None,
                    post_warmup_hook=post_warmup_hook,
                )
        finally:
            self.draft_attn_backend.token_to_kv_pool = saved_backend_pool

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        parent_list, top_scores_index, draft_tokens = (t[:raw_bs] for t in out)
        return parent_list, top_scores_index, draft_tokens

    def execute(self, forward_batch: ForwardBatch):
        self.deepep_adapter.replay()
        buffers = self.buffers

        raw_expanded_bs = forward_batch.batch_size
        raw_bs = (
            raw_expanded_bs // self.num_tokens_per_bs
            if self.topk > 1
            else raw_expanded_bs
        )
        raw_num_token = raw_expanded_bs

        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = max_num_tokens // (
                self.num_tokens_per_bs * self.num_tokens_per_bs
            )
            bs = self._pad_to_bucket(int(max_batch_size), self.capture_bs)
        else:
            bs = self._pad_to_bucket(raw_bs, self.capture_bs)

        expanded_bs = bs * self.num_tokens_per_bs
        if bs != raw_bs:
            buffers.seq_lens.fill_(self.seq_len_fill_value)
            buffers.positions.zero_()
            # Pair with seq_lens fill: padded rows must point at reserved
            # req_pool slot 0 (req_to_token[0, :] is all zeros from init).
            buffers.req_pool_indices.zero_()

        num_tokens = expanded_bs
        buffers.seq_lens[:raw_expanded_bs].copy_(forward_batch.seq_lens)
        buffers.positions[:raw_num_token].copy_(forward_batch.positions)
        if forward_batch.mrope_positions is not None:
            buffers.mrope_positions[:, :raw_num_token].copy_(
                forward_batch.mrope_positions
            )
        # `topk_p`/`topk_index` are produced by the captured seed iter.
        buffers.bonus_tokens[:raw_bs].copy_(forward_batch.spec_info.bonus_tokens)
        buffers.hidden_states[:raw_bs].copy_(forward_batch.spec_info.hidden_states)
        buffers.req_pool_indices[:raw_expanded_bs].copy_(forward_batch.req_pool_indices)

        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(expanded_bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(expanded_bs)

        if bs != raw_bs:
            forward_batch.batch_size = expanded_bs
            forward_batch.seq_lens = buffers.seq_lens[:expanded_bs]
            forward_batch.req_pool_indices = buffers.req_pool_indices[:expanded_bs]
            forward_batch.positions = buffers.positions[:num_tokens]
            if forward_batch.mrope_positions is not None:
                forward_batch.mrope_positions = buffers.mrope_positions[:, :num_tokens]

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_expanded_bs].copy_(forward_batch.seq_lens_cpu)
            forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:expanded_bs]

        self.frozen_kv_mtp_worker._init_frozen_kv_metadata_replay_cuda_graph(
            forward_batch,
            expanded_bs,
            forward_batch.seq_lens_sum
            + (expanded_bs - raw_expanded_bs) * self.seq_len_fill_value,
        )

        self.raw_bs = raw_bs
        self.bs = bs
        shape_key = self._make_graph_key(bs)
        # NVTX span: the graph bypasses `model_runner.forward`'s record_function.
        span_name = f"step[DRAFT_LOOP raw_bs={raw_bs} bs={bs} topk={self.topk}]"
        if torch.autograd._profiler_enabled():
            with torch.profiler.record_function(span_name):
                out = self._replay_graph(shape_key, forward_batch)
        else:
            out = self._replay_graph(shape_key, forward_batch)

        if bs != raw_bs:
            out = self._postprocess_output_to_raw_bs(out, raw_bs)
            forward_batch.batch_size = raw_expanded_bs
            forward_batch.positions = buffers.positions[:raw_num_token]
            forward_batch.seq_lens = buffers.seq_lens[:raw_expanded_bs]
            forward_batch.req_pool_indices = buffers.req_pool_indices[:raw_expanded_bs]
            if forward_batch.mrope_positions is not None:
                forward_batch.mrope_positions = buffers.mrope_positions[
                    :, :raw_num_token
                ]
            if forward_batch.seq_lens_cpu is not None:
                forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:raw_expanded_bs]

        return out
