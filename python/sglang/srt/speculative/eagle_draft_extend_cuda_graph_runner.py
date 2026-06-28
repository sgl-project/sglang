from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
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
    _grouped_foreach_copy_,
    get_batch_sizes_to_capture,
    model_capture_mode,
)
from sglang.srt.model_executor.runner_backend.utils import resolve_decode_backend
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.eagle_utils import get_draft_input_from_target_hidden_dim
from sglang.srt.speculative.spec_utils import fast_topk
from sglang.srt.utils import (
    is_hip,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


@dataclass
class EagleDraftExtendInputBuffers(ForwardInputBuffers):
    input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    hidden_states: Optional[torch.Tensor]
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    num_correct_drafts: torch.Tensor
    num_accept_tokens: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]


class EAGLEDraftExtendCudaGraphRunner(DecodeCudaGraphRunner):
    """EAGLE draft-extend cuda-graph runner.

    Subclasses DecodeCudaGraphRunner to inherit the outer capture
    loop + backend scaffolding. Overrides capture_one_shape,
    replay, can_run_graph for EAGLE-specific draft-extend semantics.
    """

    def __init__(
        self,
        eagle_worker: EagleDraftWorker,
        *,
        draft_extend_attn_backend=None,
        speculative_num_steps: Optional[int] = None,
    ):
        # Parse args
        self.eagle_worker = eagle_worker
        self.model_runner = model_runner = eagle_worker.draft_runner
        self.forward_mode = ForwardMode.DRAFT_EXTEND_V2

        # Fields the parent's capture() reads:
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.tp_size = model_runner.tp_size
        self.dp_size = model_runner.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.speculative_num_steps = (
            model_runner.server_args.speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.draft_extend_attn_backend = (
            draft_extend_attn_backend or eagle_worker.draft_extend_attn_backend
        )
        self.attn_backend = self.draft_extend_attn_backend

        # Disable parent paths that don't apply.
        self.compile_bs = []
        self.enable_pdmux = False
        self.record_nolora_graph = False
        self.is_dllm = False

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.capture_forward_mode = self.forward_mode
        self.capture_hidden_mode = CaptureHiddenMode.LAST

        self.capture_bs, _ = get_batch_sizes_to_capture(model_runner)
        self.padded_static_len = -1

        # Size cuda-graph buffers by num_draft_tokens (full tree width), not
        # num_steps + 1, or topk > 1 draft-extend overflows them.
        self.num_tokens_per_bs = model_runner.server_args.speculative_num_draft_tokens
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        self.draft_extend_attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        self.seq_len_fill_value = (
            self.draft_extend_attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        with torch.device(model_runner.device):
            input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            out_cache_loc = torch.ones(
                (self.max_num_token,), dtype=self._cache_loc_dtype()
            )
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)

            # Width and dtype both come from the draft `model_runner` so the
            # source stays consistent (the draft dtype matches the target dtype
            # that produced these hidden states).
            _hidden_dtype = model_runner.model_config.dtype
            _hidden_size = (
                None
                if self.eagle_worker.speculative_algorithm.is_standalone()
                else get_draft_input_from_target_hidden_dim(model_runner)
            )
            hidden_states = (
                torch.zeros(
                    (self.max_num_token, _hidden_size),
                    dtype=_hidden_dtype,
                )
                if _hidden_size is not None
                else None
            )
            self.seq_len_fill_value = (
                self.draft_extend_attn_backend.get_cuda_graph_seq_len_fill_value()
            )
            seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64
            )
            extend_seq_lens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            num_correct_drafts = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            num_accept_tokens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
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

            hot_token_id = getattr(self.eagle_worker, "hot_token_id", None)
            if hasattr(
                self.model_runner.model_config.hf_config, "draft_vocab_size"
            ):  # llama_eagle
                vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size
            elif hasattr(
                self.model_runner.model_config.hf_config, "hot_vocab_size"
            ):  # llama_eagle3
                vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size
            elif hot_token_id is not None:
                # FR-Spec: reduced vocab is injected via a late
                # json_model_override_args, so hf_config lacks it; size from the head.
                vocab_size = len(hot_token_id)
            else:
                vocab_size = self.model_runner.model_config.vocab_size

            next_token_logits_buffer = torch.zeros(
                (
                    self.max_bs * self.num_tokens_per_bs,
                    vocab_size,
                ),
                dtype=torch.float,
            )

        seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int64, device="cpu"
        )

        self.buffers = EagleDraftExtendInputBuffers(
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_positions=mrope_positions,
            hidden_states=hidden_states,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=num_accept_tokens,
            next_token_logits_buffer=next_token_logits_buffer,
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
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _replay_graph(self, shape_key, forward_batch):
        return self.backend.replay(shape_key, forward_batch)

    def _cache_loc_dtype(self):
        return torch.int64

    def _make_graph_key(self, bs, stream_idx=None, variant_label=None):
        return ShapeKey(size=bs)

    def can_run_graph(self, forward_batch: ForwardBatch):
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
            self.backend.can_run(forward_batch, cuda_graph_bs)
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
        bs = size
        buffers = self.buffers
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = buffers.input_ids[:num_tokens]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        extend_seq_lens = buffers.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        hidden_states = (
            buffers.hidden_states[:num_tokens]
            if buffers.hidden_states is not None
            else None
        )
        num_correct_drafts = buffers.num_correct_drafts[:bs]
        num_accept_tokens = buffers.num_accept_tokens[:bs]
        next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]

        # pruned_states = num_tokens (all tokens)
        num_tokens_for_logprob = num_tokens

        if self.require_mlp_tp_gather:
            global_num_tokens_cpu = [num_tokens] * self.dp_size
        elif self.require_attn_tp_gather:
            global_num_tokens_cpu = [num_tokens]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    global_num_tokens_cpu,
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens_for_logprob] * len(global_num_tokens_cpu),
                    dtype=torch.int32,
                    device=buffers.input_ids.device,
                )
            )
        else:
            global_dp_buffer_len = None

        spec_info = EagleDraftExtendInput(
            hidden_states=hidden_states,
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=num_accept_tokens,
            # Padded tree width per req; drives the constant qo layout.
            num_tokens_per_req=self.num_tokens_per_bs,
        )

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
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            padded_static_len=self.padded_static_len,
        )

        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
                global_num_tokens_cpu,
            )
            set_is_extend_in_batch(False)

            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
            # ROCm's argmax tie-breaks differently from CUDA's softmax+max
            # path on FP8 logits, which corrupts MTP draft selection on AMD.
            # Keep the fastpath CUDA-only.
            if self.topk == 1 and not _is_hip:
                ret.topk_index = torch.argmax(
                    ret.next_token_logits, dim=-1, keepdim=True
                )
                ret.topk_p = torch.ones_like(ret.topk_index, dtype=torch.float32)
            else:
                probs = torch.softmax(ret.next_token_logits, dim=-1)
                ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)

            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        with forward_context(
            ForwardContext(attn_backend=self.draft_extend_attn_backend)
        ):
            self.draft_extend_attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=True
            )
            self.deepep_adapter.capture(is_extend_in_batch=True)
            canary_ctx = (
                c.with_active_single_forward_manager(0)
                if (c := self.model_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with canary_ctx:
                shape_key = self._make_graph_key(bs)
                self.backend.capture_one(
                    shape_key,
                    run_once,
                    dummies=None,
                    post_warmup_hook=getattr(
                        self.draft_extend_attn_backend,
                        "on_after_cuda_graph_warmup",
                        None,
                    ),
                )

    def execute(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        self.deepep_adapter.replay()
        buffers = self.buffers

        raw_bs = forward_batch.batch_size
        num_tokens = forward_batch.input_ids.shape[0]
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max_num_tokens
            )
            bs = self._pad_to_bucket(int(max_batch_size), self.capture_bs)
        else:
            bs = self._pad_to_bucket(raw_bs, self.capture_bs)

        if bs * self.num_tokens_per_bs != num_tokens:
            buffers.seq_lens.fill_(self.seq_len_fill_value)
            buffers.out_cache_loc.zero_()
            buffers.positions.zero_()
            # Pair with seq_lens fill: padded rows must point at reserved
            # req_pool slot 0 (req_to_token[0, :] is all zeros from init).
            buffers.req_pool_indices.zero_()
            buffers.num_correct_drafts.fill_(self.num_tokens_per_bs)
            buffers.num_accept_tokens.fill_(self.num_tokens_per_bs)
            buffers.extend_seq_lens.fill_(self.num_tokens_per_bs)

        # Batch the small per-field device copies into a grouped foreach copy
        # (one foreach call per dtype pair) to cut launch overhead. hidden_states
        # is handled separately below (see note), and seq_lens_cpu is handled
        # further down since it lives on host.
        copy_dsts = [
            buffers.input_ids[:num_tokens],
            buffers.seq_lens[:raw_bs],
            buffers.out_cache_loc[:num_tokens],
            buffers.positions[:num_tokens],
            buffers.req_pool_indices[:raw_bs],
        ]
        copy_srcs = [
            forward_batch.input_ids,
            forward_batch.seq_lens,
            forward_batch.out_cache_loc,
            forward_batch.positions,
            forward_batch.req_pool_indices,
        ]
        if forward_batch.extend_seq_lens is not None:
            copy_dsts.append(buffers.extend_seq_lens[:raw_bs])
            copy_srcs.append(forward_batch.extend_seq_lens)
        else:
            buffers.extend_seq_lens[:raw_bs].fill_(self.num_tokens_per_bs)
        if forward_batch.spec_info.num_correct_drafts is not None:
            copy_dsts.append(buffers.num_correct_drafts[:raw_bs])
            copy_srcs.append(forward_batch.spec_info.num_correct_drafts)
            copy_dsts.append(buffers.num_accept_tokens[:raw_bs])
            copy_srcs.append(forward_batch.spec_info.num_accept_tokens)
        _grouped_foreach_copy_(copy_dsts, copy_srcs)

        # hidden_states is large + contiguous: copy_() uses the cudaMemcpyAsync
        # DMA engine; foreach would force the ~3x slower compute-kernel copy.
        if (
            buffers.hidden_states is not None
            and forward_batch.spec_info.hidden_states is not None
            and forward_batch.spec_info.hidden_states.shape[1]
            == buffers.hidden_states.shape[1]
        ):
            buffers.hidden_states[:num_tokens].copy_(
                forward_batch.spec_info.hidden_states
            )

        # TODO(ch-wan): support num_token_non_padded
        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(bs * self.num_tokens_per_bs)

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

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
        forward_batch.spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[:bs]

        if bs != raw_bs:
            forward_batch.spec_info.positions = buffers.positions[:num_tokens]
            forward_batch.spec_info.num_correct_drafts = buffers.num_correct_drafts[:bs]
            forward_batch.spec_info.num_accept_tokens = buffers.num_accept_tokens[:bs]

        from types import SimpleNamespace

        seq_lens_sum = forward_batch.seq_lens_sum
        if seq_lens_sum is not None:
            seq_lens_sum = seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value
        fb_view = SimpleNamespace(
            batch_size=bs,
            forward_mode=self.forward_mode,
            input_ids=getattr(forward_batch, "input_ids", None),
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=buffers.seq_lens_cpu,
            encoder_lens=None,
            out_cache_loc=forward_batch.out_cache_loc,
            spec_info=forward_batch.spec_info,
        )
        self.draft_extend_attn_backend.init_forward_metadata_out_graph(fb_view)

        # Snapshot built -- the forward is done reading the shared pool. Publish
        # a read-done event the scheduler's WAR barrier waits on.
        read_done = self.device_module.Event()
        read_done.record()
        self.model_runner.war_fastpath_read_done_event = read_done

        self.raw_bs = raw_bs
        self.bs = bs
        shape_key = self._make_graph_key(bs)
        timer_ctx = (
            self.model_runner.device_timer.wrap(
                metadata={"category": "eagle_draft_extend"}
            )
            if self.model_runner.device_timer
            else contextlib.nullcontext()
        )
        with timer_ctx:
            out = self._replay_graph(shape_key, forward_batch)

        out = LogitsProcessorOutput(
            next_token_logits=out.next_token_logits[:num_tokens],
            hidden_states=out.hidden_states[:num_tokens],
        )
        return out
