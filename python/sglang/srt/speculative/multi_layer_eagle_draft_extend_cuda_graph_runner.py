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

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    ShapeKey,
    get_batch_sizes_to_capture,
    model_capture_mode,
)
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    maybe_flashinfer_autotune_speculative_draft,
)
from sglang.srt.model_executor.runner_backend.utils import resolve_decode_backend
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.runtime_context import get_flags
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.eagle_utils import get_draft_input_from_target_hidden_dim
from sglang.srt.speculative.multi_layer_eagle_utils import (
    fill_draft_extend_prepare_buffers_triton,
    rotate_input_ids,
    wide_row_softmax_triton,
)
from sglang.srt.speculative.spec_utils import (
    fast_sample,
    fast_topk,
    resolve_num_tokens_per_req,
)
from sglang.srt.utils import (
    get_available_gpu_memory,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
        MultiLayerEagleDraftWorker,
    )


logger = logging.getLogger(__name__)


@dataclass
class MultiLayerEagleDraftExtendInputBuffers(ForwardInputBuffers):
    """A single persistent buffer set shared by every MTP draft step."""

    input_ids: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    num_correct_drafts: torch.Tensor
    num_accept_tokens: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_start_loc: torch.Tensor
    # Flat index (into the token dimension) of each request's last accepted
    # token. Used both by the in-graph top-k gather and by the worker's
    # per-step input_ids rotation.
    select_index: torch.Tensor
    mrope_positions: torch.Tensor
    hidden_states: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]
    # Rejection sampling with the single-CG runner only, else None (the presence
    # of draft_probs selects the in-graph proposal branch in _run_step_body).
    temperatures: Optional[torch.Tensor]
    draft_probs: Optional[torch.Tensor]


class MultiLayerEagleDraftExtendCudaGraphRunner(DecodeCudaGraphRunner):
    """Per-step multi-layer EAGLE draft-extend runner.

    Subclasses DecodeCudaGraphRunner. All steps share a single buffer set
    owned by the composite MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
    so initialization is split: __init__ does basic field setup, and
    init_buffers_and_capture (called by the composite once the shared buffers
    exist) attaches them and runs capture.
    """

    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker, step: int):
        # Parse args
        self.step = step
        self.eagle_worker = eagle_worker
        self.model_runner = model_runner = eagle_worker.mtp_model_runner(self.step)
        self.forward_mode = ForwardMode.DRAFT_EXTEND_V2

        # Fields the parent's capture() reads:
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.tp_size = model_runner.ps.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.enable_torch_compile = get_flags().capture.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_pdmux = model_runner.server_args.enable_pdmux
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.attn_backend = self.eagle_worker.draft_extend_attn_backend_list[self.step]
        self.metadata_captured_in_graph = (
            self.attn_backend.draft_extend_metadata_captured_in_graph()
        )
        # Gathered-buffer (DP) modes size the logprob gather for all-row
        # logits, so they keep the unpruned lm_head path.
        self.prune_draft_extend_logits = not self.require_gathered_buffer

        # Disable parent paths that don't apply.
        self.compile_bs = []
        self.record_nolora_graph = False
        self.is_dllm = False

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.capture_forward_mode = self.forward_mode
        self.capture_hidden_mode = CaptureHiddenMode.FULL

        self.capture_bs, _ = get_batch_sizes_to_capture(model_runner)

        # Fixed window: every step extends each request by the same number of
        # tokens, which lets all steps share one buffer set.
        self.num_front_tokens = eagle_worker.draft_extend_num_front_tokens
        self.captured_req_width = (
            resolve_num_tokens_per_req(
                phase="draft_extend", server_args=model_runner.server_args
            )
            + self.num_front_tokens
        )
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.captured_req_width
        self.extend_seq_lens_cpu = [self.captured_req_width] * self.max_bs

        self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].init_cuda_graph_state(self.max_bs, self.max_num_token)
        self.seq_len_fill_value = self.eagle_worker.draft_extend_attn_backend_list[
            self.step
        ].get_cuda_graph_seq_len_fill_value()

    def init_buffers_and_capture(self, buffers: MultiLayerEagleDraftExtendInputBuffers):
        """Attach the shared buffer set and capture this step's graphs."""
        self.buffers = buffers

        if self.enable_torch_compile:
            set_torch_compile_config()

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

    def _make_graph_key(self, bs, stream_idx=None, variant_label=None):
        return ShapeKey(size=bs)

    def can_run_graph(self, forward_batch: ForwardBatch):
        # Uniform-width replay invariant: the batch's actual per-request width
        # must match this runner's capture width; anything else falls back to
        # eager. (Unset widths pass: not every path fills the field yet.)
        spec_info = forward_batch.spec_info
        if (
            spec_info is not None
            and spec_info.num_tokens_per_req > 0
            and spec_info.num_tokens_per_req != self.captured_req_width
        ):
            return False

        if self.require_mlp_tp_gather:
            # Raw sync values are per-rank request counts on decode-family rounds.
            cuda_graph_bs = max(forward_batch.original_global_num_tokens_cpu)
        else:
            cuda_graph_bs = forward_batch.seq_lens.numel()

        is_bs_supported = (
            self.backend.can_run(forward_batch, self._make_graph_key(cuda_graph_bs))
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported

    def get_forward_batch(self, bs: int) -> ForwardBatch:
        buffers = self.buffers
        num_tokens = bs * self.captured_req_width

        input_ids = buffers.input_ids[:num_tokens]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        extend_seq_lens = buffers.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        extend_start_loc = buffers.extend_start_loc[:bs]
        num_correct_drafts = buffers.num_correct_drafts[:bs]
        num_accept_tokens = buffers.num_accept_tokens[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        hidden_states = buffers.hidden_states[:num_tokens]
        if self.prune_draft_extend_logits:
            next_token_logits_buffer = buffers.next_token_logits_buffer[:bs]
        else:
            next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]

        if self.require_mlp_tp_gather:
            global_num_tokens_cpu = [num_tokens] * self.dp_size
            global_num_tokens_for_logprob_cpu = [num_tokens] * self.dp_size
        elif self.require_attn_tp_gather:
            global_num_tokens_cpu = [num_tokens]
            # DRAFT_EXTEND_V2 produces logits for all tokens, not bs (see mlp branch above)
            global_num_tokens_for_logprob_cpu = [num_tokens]
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
                    global_num_tokens_for_logprob_cpu,
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
            num_front_tokens=self.num_front_tokens,
        )
        spec_info.num_tokens_per_req = self.captured_req_width
        spec_info.positions = None
        if self.prune_draft_extend_logits:
            spec_info.select_index = buffers.select_index[:bs]

        capture_mode = (
            CaptureHiddenMode.NULL
            if self.model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )

        # Forward batch
        forward_batch = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            global_num_tokens_cpu=global_num_tokens_cpu,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_start_loc=extend_start_loc,
            extend_num_tokens=self.captured_req_width * bs,
            num_token_non_padded_cpu=self.captured_req_width * bs,
            return_hidden_states_before_norm=True,
        )
        return forward_batch

    def _postprocess_forward_batch(self, forward_batch: ForwardBatch, bs: int):
        """Hook for subclasses to mutate the captured forward batch."""
        return forward_batch

    def _select_step_logits(self, ret, bs: int):
        """Each request's last-accepted-row logits, [bs, vocab]."""
        if self.prune_draft_extend_logits:
            return ret.next_token_logits
        return ret.next_token_logits[self.buffers.select_index[:bs]]

    def _compute_topk(self, ret, bs: int):
        """Compute top-k on the last accepted token's logits and attach it to
        ``ret``. Overridable so distributed (vocab-sharded) builds can plug in
        an all-reduce-aware sampler."""
        probs = torch.softmax(self._select_step_logits(ret, bs), dim=-1)
        ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)

    def _sample_draft_proposal(self, ret, bs: int):
        """In-graph Leviathan proposal (single-CG runner + rejection sampling):
        q = softmax(logits / T) written straight into this step's draft_probs
        slot, then X ~ q. The accept test coin*q(X) < p(X) is unbiased only if
        q is exactly the distribution X was drawn from -- q here IS the
        stashed tensor."""
        buffers = self.buffers
        probs = wide_row_softmax_triton(
            self._select_step_logits(ret, bs),
            buffers.temperatures[:bs],
            buffers.draft_probs[:bs, self.step],
        )
        ret.topk_p, ret.topk_index = fast_sample(probs, num_samples=1)

    def _run_step_body(self, forward_batch: ForwardBatch, num_tokens: int, bs: int):
        """One draft step's body: model forward + chain-hidden write + top-k.
        Shared by the per-step capture and the single-CG multi-step capture."""
        buffers = self.buffers
        # Clean intermediate result cache for DP attention
        forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
        set_dp_buffer_len(
            forward_batch.global_dp_buffer_len,
            num_tokens,
            forward_batch.dp_padding_mode.is_max_len(),
            forward_batch.global_num_tokens_cpu,
        )
        set_is_extend_in_batch(False)

        output_cache_loc_backup = forward_batch.out_cache_loc
        hidden_states_backup = forward_batch.spec_info.hidden_states

        ret = self.model_runner.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )

        if self.eagle_worker.chain_mtp_hidden_states and ret.hidden_states is not None:
            buffers.hidden_states[:num_tokens].copy_(ret.hidden_states[:num_tokens])

        if buffers.draft_probs is not None:
            self._sample_draft_proposal(ret, bs)
        else:
            self._compute_topk(ret, bs)

        forward_batch.out_cache_loc = output_cache_loc_backup
        forward_batch.spec_info.hidden_states = hidden_states_backup
        return ret

    def capture_one_shape(
        self,
        size: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
    ):
        bs = size

        num_tokens = bs * self.captured_req_width
        forward_batch = self.get_forward_batch(bs)
        forward_batch = self._postprocess_forward_batch(forward_batch, bs)
        attn_backend = self.eagle_worker.draft_extend_attn_backend_list[self.step]

        def run_once():
            attn_backend.init_forward_metadata_in_graph(forward_batch)
            return self._run_step_body(forward_batch, num_tokens, bs)

        with forward_context(ForwardContext(attn_backend=attn_backend)):
            attn_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
            self.deepep_adapter.capture(is_extend_in_batch=True)
            shape_key = self._make_graph_key(bs)
            post_warmup_hook = getattr(
                self.attn_backend, "on_after_cuda_graph_warmup", None
            )
            maybe_flashinfer_autotune_speculative_draft(
                self,
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

    def replay(
        self,
        bs: int,
        seq_lens_sum: Optional[int],
        spec_info: EagleDraftExtendInput,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Init this step's attention metadata for the prepared bucket and
        replay its graph. Buffers must already be populated by the composite
        runner's ``prepare`` (step 0) or by the previous step's in-graph chain
        write + worker-side rotation (steps > 0). Backends whose in-graph
        metadata update is complete recompute it inside the replayed graph;
        others get the eager out-graph rebuild here."""
        self.deepep_adapter.replay()
        buffers = self.buffers
        num_tokens = bs * self.captured_req_width

        fb_view = SimpleNamespace(
            batch_size=bs,
            forward_mode=self.forward_mode,
            input_ids=buffers.input_ids[:num_tokens],
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            encoder_lens=None,
            # per-step write target; out_cache_loc is frozen at prepare() time.
            out_cache_loc=buffers.out_cache_loc[:num_tokens],
            spec_info=spec_info,
        )
        if not self.metadata_captured_in_graph:
            self.eagle_worker.draft_extend_attn_backend_list[
                self.step
            ].init_forward_metadata_out_graph(fb_view)

        self.bs = bs
        shape_key = self._make_graph_key(bs)
        return self._replay_graph(shape_key, fb_view)


class MultiLayerEagleMultiStepDraftExtendCudaGraphRunner:
    """Owns one shared buffer set and the per-step runners.

    Usage from the worker::

        runner.prepare(forward_batch)
        for step in range(num_steps):
            _, topk_p, topk_index = runner.replay(step)
            if step < num_steps - 1:
                rotate_input_ids(...)  # advance the draft chain

    Not itself a DecodeCudaGraphRunner -- it only routes work to the per-step
    runners.
    """

    rotates_in_graph = False

    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker):
        self.eagle_worker = eagle_worker
        self.device = eagle_worker.device
        self.gpu_id = eagle_worker.gpu_id
        self.speculative_num_steps = eagle_worker.speculative_num_steps
        self.draft_extend_attn_backend_list = (
            eagle_worker.draft_extend_attn_backend_list
        )

        self.runners: List[Optional[MultiLayerEagleDraftExtendCudaGraphRunner]] = []
        self.seq_len_fill_value = 1
        self.max_bs = 1
        self.captured_req_width = 1
        self.num_front_tokens = 0
        self.prune_draft_extend_logits = False

        self._init_and_capture()

    def _create_runner(self, step: int) -> MultiLayerEagleDraftExtendCudaGraphRunner:
        return MultiLayerEagleDraftExtendCudaGraphRunner(self.eagle_worker, step)

    def _capture_context(self, step: int):
        """Context manager active while capturing ``step``'s graphs. Subclasses
        can use it e.g. to temporarily expose a sharded local vocab size."""
        return contextlib.nullcontext()

    def _create_runners(self):
        """Construct per-step runners (each initializes its own attn cuda graph
        state; all share the same fixed window size) and mirror their shared
        shape attributes onto self."""
        self.runners = []
        for step in range(self.speculative_num_steps):
            if self.draft_extend_attn_backend_list[step]:
                runner = self._create_runner(step)
                self.runners.append(runner)
                self.seq_len_fill_value = runner.seq_len_fill_value
                self.max_bs = runner.max_bs
                self.captured_req_width = runner.captured_req_width
                self.num_front_tokens = runner.num_front_tokens
                self.capture_bs = runner.capture_bs
                self.require_gathered_buffer = runner.require_gathered_buffer
                self.require_mlp_tp_gather = runner.require_mlp_tp_gather
                self.require_mlp_sync = runner.require_mlp_sync
                self.disable_padding = runner.disable_padding
                self.prune_draft_extend_logits = runner.prune_draft_extend_logits
            else:
                self.runners.append(None)

    def _on_runners_created(self):
        """Hook called after all per-step runners exist but before buffers are
        allocated/captured (e.g. to allocate shared sconv buffers)."""

    def _cuda_graph_disabled(self) -> bool:
        return check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)

    def _init_and_capture(self):
        if self._cuda_graph_disabled():
            self.runners = [None] * self.speculative_num_steps
            return

        self._create_runners()

        self._on_runners_created()

        # 2. Allocate the single shared buffer set and capture each step in
        #    reverse order.
        self.buffers = self._allocate_buffers()
        for step in range(self.speculative_num_steps - 1, -1, -1):
            if self.runners[step] is not None:
                tic = time.perf_counter()
                before_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    f"Capture draft extend CUDA graph begin. step={step}, "
                    f"avail mem={before_mem:.2f} GB"
                )

                with self._capture_context(step):
                    self.runners[step].init_buffers_and_capture(self.buffers)

                after_mem = get_available_gpu_memory(self.device, self.gpu_id)
                logger.info(
                    "Capture draft extend CUDA graph end. "
                    f"step={step}, elapsed={time.perf_counter() - tic:.2f} s, "
                    f"mem usage={(before_mem - after_mem):.2f} GB, "
                    f"avail mem={after_mem:.2f} GB."
                )

    def _vocab_size(self) -> int:
        model_runner = self.eagle_worker.mtp_model_runner(0)
        if hasattr(model_runner.model_config.hf_config, "draft_vocab_size"):
            return model_runner.model_config.hf_config.draft_vocab_size
        if hasattr(model_runner.model_config.hf_config, "hot_vocab_size"):
            return model_runner.model_config.hf_config.hot_vocab_size
        return model_runner.model_config.vocab_size

    def _allocate_buffers(self) -> MultiLayerEagleDraftExtendInputBuffers:
        runner = next(r for r in self.runners if r is not None)
        model_runner = runner.model_runner
        max_bs = self.max_bs
        num_tokens_per_req = self.captured_req_width
        max_num_token = max_bs * num_tokens_per_req
        hidden_size = get_draft_input_from_target_hidden_dim(model_runner)
        dtype = model_runner.model_config.dtype
        vocab_size = self._vocab_size()

        seq_lens_cpu = torch.full((max_bs,), self.seq_len_fill_value, dtype=torch.int32)

        with torch.device(self.device):
            input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
            out_cache_loc = torch.ones((max_num_token,), dtype=torch.int64)
            positions = torch.zeros((max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, max_num_token), dtype=torch.int64)
            hidden_states = torch.zeros((max_num_token, hidden_size), dtype=dtype)

            seq_lens = torch.full((max_bs,), self.seq_len_fill_value, dtype=torch.int32)
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int64)
            num_correct_drafts = torch.full((max_bs,), 1, dtype=torch.int32)
            num_accept_tokens = torch.full((max_bs,), 1, dtype=torch.int32)

            # Fixed window: every request extends by exactly num_tokens_per_req
            # tokens, and start locs are a constant arange.
            extend_seq_lens = torch.full(
                (max_bs,), num_tokens_per_req, dtype=torch.int32
            )
            extend_start_loc = torch.arange(
                0, max_num_token, step=num_tokens_per_req, dtype=torch.int32
            )
            select_index = torch.zeros((max_bs,), dtype=torch.int64)

            next_token_logits_buffer = torch.zeros(
                (
                    max_bs if self.prune_draft_extend_logits else max_num_token,
                    vocab_size,
                ),
                dtype=torch.float,
            )

            if (
                self.rotates_in_graph
                and self.eagle_worker.use_rejection_sampling
                and self.eagle_worker.topk == 1
            ):
                temperatures = torch.ones((max_bs, 1), dtype=torch.float)
                draft_probs = torch.empty(
                    (max_bs, self.speculative_num_steps, vocab_size),
                    dtype=torch.float,
                )
            else:
                temperatures = None
                draft_probs = None

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    dp_size = runner.dp_size
                    global_num_tokens_gpu = torch.zeros((dp_size,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (dp_size,), dtype=torch.int32
                    )
                else:
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

        return MultiLayerEagleDraftExtendInputBuffers(
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
            select_index=select_index,
            mrope_positions=mrope_positions,
            hidden_states=hidden_states,
            next_token_logits_buffer=next_token_logits_buffer,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            temperatures=temperatures,
            draft_probs=draft_probs,
        )

    def _prepare_extra(self, forward_batch: ForwardBatch) -> None:
        """Hook for subclasses to populate extra per-call buffers (e.g. sconv)."""

    def prepare(self, forward_batch: ForwardBatch):
        """Populate the shared buffers once from ``forward_batch`` and bucketize
        the batch size. Subsequent ``replay(step)`` calls reuse this state."""
        buffers = self.buffers
        raw_bs = forward_batch.batch_size
        num_tokens = raw_bs * self.captured_req_width

        # Bucketize to a captured batch size (padding the tail).
        if self.require_mlp_tp_gather:
            max_batch_size = max(forward_batch.original_global_num_tokens_cpu)
            bs = self.get_runner(0)._pad_to_bucket(int(max_batch_size), self.capture_bs)
        else:
            bs = self.get_runner(0)._pad_to_bucket(raw_bs, self.capture_bs)

        fill_draft_extend_prepare_buffers_triton(
            buffers.input_ids,
            buffers.positions,
            buffers.out_cache_loc,
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch.out_cache_loc,
            buffers.seq_lens,
            buffers.req_pool_indices,
            buffers.num_correct_drafts,
            buffers.num_accept_tokens,
            buffers.select_index,
            buffers.temperatures,
            forward_batch.seq_lens,
            forward_batch.req_pool_indices,
            forward_batch.spec_info.num_correct_drafts,
            forward_batch.spec_info.num_accept_tokens,
            (
                forward_batch.sampling_info.temperatures
                if buffers.temperatures is not None
                else None
            ),
            buffers.hidden_states,
            (
                forward_batch.spec_info.hidden_states
                if forward_batch.spec_info.hidden_states.shape[1]
                == buffers.hidden_states.shape[1]
                else None
            ),
            buffers.global_num_tokens_gpu if self.require_gathered_buffer else None,
            (
                buffers.global_num_tokens_for_logprob_gpu
                if self.require_gathered_buffer
                else None
            ),
            raw_bs,
            bs,
            self.captured_req_width,
            self.num_front_tokens,
            self.seq_len_fill_value,
        )

        # Refresh the host mirror only when published; hand replay None
        # otherwise so no consumer reads a stale buffer.
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
            self.seq_lens_cpu = buffers.seq_lens_cpu
        else:
            self.seq_lens_cpu = None

        # Reusable spec_info for per-step attention metadata.
        padded_num_tokens = bs * self.captured_req_width
        spec_info = EagleDraftExtendInput(
            hidden_states=buffers.hidden_states[:padded_num_tokens],
            num_correct_drafts=buffers.num_correct_drafts[:bs],
            num_accept_tokens=buffers.num_accept_tokens[:bs],
            num_front_tokens=self.num_front_tokens,
        )
        # Actual width of the captured forward == static width by construction.
        spec_info.num_tokens_per_req = self.captured_req_width
        spec_info.num_tokens_for_logprob_per_req = 1
        spec_info.positions = buffers.positions[:padded_num_tokens]
        spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[:bs]
        self._replay_spec_info = spec_info

        self.raw_bs = raw_bs
        self.bs = bs
        self.raw_num_tokens = num_tokens
        seq_lens_sum = forward_batch.seq_lens_sum
        if seq_lens_sum is not None:
            seq_lens_sum = seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value
        self.seq_lens_sum = seq_lens_sum

        self._prepare_extra(forward_batch)

    def replay(self, step: int):
        """Replay ``step``'s graph at the prepared bucket. Returns
        ``(LogitsProcessorOutput, topk_p, topk_index)`` sliced to the real
        batch size."""
        runner = self.runners[step]
        runner.raw_bs = self.raw_bs
        out = runner.replay(
            self.bs, self.seq_lens_sum, self._replay_spec_info, self.seq_lens_cpu
        )
        raw_bs = self.raw_bs
        raw_num_tokens = self.raw_num_tokens
        num_logit_rows = raw_bs if self.prune_draft_extend_logits else raw_num_tokens
        logits_output = LogitsProcessorOutput(
            next_token_logits=out.next_token_logits[:num_logit_rows],
            hidden_states=(
                out.hidden_states[:raw_num_tokens]
                if out.hidden_states is not None
                else None
            ),
        )
        return (
            logits_output,
            out.topk_p[:raw_bs],
            out.topk_index[:raw_bs],
        )

    def clone_draft_probs(self) -> torch.Tensor:
        """Materialize the in-graph-written proposal q [raw_bs, num_steps, vocab]
        after replay; the clone must land before a later replay rewrites the
        buffer (spec_info.draft_probs outlives this decode step on paused batches)."""
        return self.buffers.draft_probs[: self.raw_bs].clone()

    def get_runner(self, step):
        return self.runners[step]

    def get_last_runner(self):
        return self.runners[-1] if self.runners else None

    def can_run_graph(self, forward_batch):
        return self.runners[0].can_run_graph(forward_batch)


class OneGraphMultiLayerEagleMultiStepDraftExtendCudaGraphRunner(
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
):
    """Single-CG variant (SGLANG_ENABLE_SINGLE_CG_DRAFT): captures all draft steps'
    forwards + the inter-step input_ids rotation in ONE graph per bucket, instead
    of one graph per step. The worker drops its per-step rotation (rotates_in_graph).

    Each step's replay metadata is emitted in-graph via
    init_forward_metadata_in_graph (no Python may run between
    captured steps). seq_lens / req_pool_indices / extend_seq_lens are chain-constant
    (only input_ids rotates), so per-step in-graph metadata is correct.

    Rejection sampling is supported by sampling X ~ q inside the graph
    (_sample_draft_proposal, selected by the draft_probs buffer's presence):
    the captured rotation then carries the sampled token, prepare() stages the
    temperatures sampling_info cannot deliver in-graph, and the worker clones
    the per-step q off buffers.draft_probs after replay. torch.multinomial
    draws through the graph-registered Philox generator, so each replay gets
    fresh coins.
    """

    rotates_in_graph = True

    def _init_and_capture(self):
        if self._cuda_graph_disabled():
            self.runners = [None] * self.speculative_num_steps
            return

        self._create_runners()

        self._on_runners_created()
        self.buffers = self._allocate_buffers()

        # No per-step capture: all steps are captured in one graph below.
        for r in self.runners:
            if r is not None:
                r.buffers = self.buffers
                if r.enable_torch_compile:
                    set_torch_compile_config()
                r.backend = resolve_decode_backend(r)

        runners = [r for r in self.runners if r is not None]
        first = runners[0]
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            "Capture single-CG draft extend begin. This can take several minutes. "
            f"avail mem={before_mem:.2f} GB"
        )
        try:
            with model_capture_mode():
                for bs in reversed(first.capture_bs):
                    self._capture_one_graph(bs, runners)
        except RuntimeError as e:
            raise Exception(
                f"Capture single-CG draft extend failed: {e}\n"
                f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            "Capture single-CG draft extend end. "
            f"elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_mem - after_mem):.2f} GB, avail mem={after_mem:.2f} GB."
        )

    def _capture_one_graph(self, bs: int, runners):
        buffers = self.buffers
        n = len(runners)

        items = []
        for r in runners:
            num_tokens = bs * r.captured_req_width
            forward_batch = r.get_forward_batch(bs)
            forward_batch = r._postprocess_forward_batch(forward_batch, bs)
            attn_backend = self.draft_extend_attn_backend_list[r.step]
            with forward_context(ForwardContext(attn_backend=attn_backend)):
                attn_backend.init_forward_metadata_out_graph(
                    forward_batch, in_capture=True
                )
                r.deepep_adapter.capture(is_extend_in_batch=True)
            items.append((r, forward_batch, num_tokens, attn_backend))

        def multi_step_fn():
            outs = []
            for i, (r, forward_batch, num_tokens, attn_backend) in enumerate(items):
                attn_backend.init_forward_metadata_in_graph(forward_batch)
                with forward_context(ForwardContext(attn_backend=attn_backend)):
                    ret = r._run_step_body(forward_batch, num_tokens, bs)
                outs.append(ret)
                if i < n - 1:
                    rotate_input_ids(
                        buffers.input_ids[: bs * self.captured_req_width],
                        buffers.extend_start_loc[:bs],
                        buffers.extend_seq_lens[:bs],
                        ret.topk_index,
                        buffers.select_index[:bs],
                    )
            return outs

        first = runners[0]
        # capture_one's warmup runs multi_step_fn, whose in-graph rotation mutates
        # input_ids/hidden_states/select_index; snapshot and restore OUTSIDE the
        # graph (an in-graph reset would clobber prepare()'s runtime writes).
        input_ids_orig = buffers.input_ids.clone()
        hidden_states_orig = buffers.hidden_states.clone()
        select_index_orig = buffers.select_index.clone()

        shape_key = first._make_graph_key(bs)
        first.backend.capture_one(
            shape_key,
            multi_step_fn,
            dummies=None,
            post_warmup_hook=getattr(
                first.attn_backend, "on_after_cuda_graph_warmup", None
            ),
        )

        buffers.input_ids.copy_(input_ids_orig)
        buffers.hidden_states.copy_(hidden_states_orig)
        buffers.select_index.copy_(select_index_orig)

    def replay(self, step: int):
        """Replays the one graph on step 0 and serves the rest from cache. The
        first tuple element is the step's RAW (unsliced) LogitsProcessorOutput;
        the single-CG worker path never consumes it."""
        if step == 0:
            first = self.runners[0]
            for r in self.runners:
                if r is not None:
                    r.deepep_adapter.replay()
            shape_key = first._make_graph_key(self.bs)
            outs = first.backend.replay(shape_key, self._replay_spec_info)
            raw_bs = self.raw_bs
            self._cached = {}
            non_null = [r for r in self.runners if r is not None]
            for r, out in zip(non_null, outs):
                self._cached[r.step] = (
                    out,
                    out.topk_p[:raw_bs],
                    out.topk_index[:raw_bs],
                )
        return self._cached[step]
