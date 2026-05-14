# Copyright 2023-2026 SGLang Team
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
"""PrefillCudaGraphRunner — runs the EXTEND phase under a pluggable backend.

Backend selection comes from ``cuda_graph_mode[Phase.PREFILL]``:
  - ``"tc_piecewise"``     — default, ``TcPiecewiseCudaGraphBackend``: torch.compile
                      wraps the model; per-shape graphs live in
                      torch.compile's internal cache. Multi-batch supported.
  - ``"breakable"`` — ``BreakableCudaGraphBackend``: segmented capture (no
                      torch.compile). Captures with bs=1; rejects multi-req
                      prefill in ``can_run``.
  - ``"full"``      — rejected at config validation; not supported for prefill.
  - ``"disabled"``  — handled at the model_runner level — runner not
                      constructed.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Union

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.model_executor.cuda_graph_backend.factory import (
    resolve_prefill_backend,
)
from sglang.srt.model_executor.cuda_graph_backend_utils.tc_piecewise_cuda_graph import (
    set_forward_context,
)
from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
    BaseCudaGraphRunner,
    freeze_gc,
)
from sglang.srt.model_executor.cuda_graph_runner_utils.buffers import (
    PrefillInputBuffers,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.utils import get_available_gpu_memory, is_npu, log_info_on_rank0

# Suppress Dynamo warning about tracing through lru_cache-wrapped functions.
warnings.filterwarnings("ignore", message=".*lru_cache.*", module="torch._dynamo")
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class PrefillCudaGraphRunner(BaseCudaGraphRunner):
    """Prefill-phase CUDA graph runner.

    Owns: ``PrefillInputBuffers``, capture-num-tokens list, attention layers
    snapshot, and the pluggable ``self.backend``. The backend handles capture
    + replay mechanics; this runner handles dummy ForwardBatch construction,
    buffer population, attention metadata init, and output slicing.
    """

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__(model_runner)
        # --- core state ------------------------------------------------
        self.quant_config = getattr(self.model_runner.model, "quant_config", None)
        self.is_multimodal = model_runner.is_multimodal
        # Classification/reward forwards branch on return_pooled_hidden_states;
        # capture must use the same flag value as replay for those models.
        self.capture_return_pooled_hidden_states = not model_runner.is_generation

        # --- bucket sizes ---------------------------------------------
        capture_tokens = model_runner.server_args.piecewise_cuda_graph_tokens
        assert capture_tokens is not None, "piecewise_cuda_graph_tokens is not set"
        self.capture_num_tokens = sorted(capture_tokens)
        self.max_num_tokens = (
            max(self.capture_num_tokens) if self.capture_num_tokens else 8192
        )
        self.max_bs = model_runner.req_to_token_pool.size

        log_info_on_rank0(
            logger, f"Capture cuda graph num tokens {self.capture_num_tokens}"
        )

        self.capture_forward_mode = ForwardMode.EXTEND
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        self.mamba_track_enabled = self._is_mamba_track_enabled()

        # --- buffers ---------------------------------------------------
        self.buffers: PrefillInputBuffers = PrefillInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_tokens=self.max_num_tokens,
            cache_loc_dtype=self._cache_loc_dtype(),
            is_hybrid_swa=model_runner.is_hybrid_swa,
            is_multimodal=self.is_multimodal,
            hidden_size=self.model_runner.model_config.hidden_size,
            dtype=self.model_runner.dtype,
            enable_mamba_track=self.mamba_track_enabled,
        )
        self.buffers.share_buffers()

        self.attention_layers = self.model_runner.attention_layers
        self.moe_layers = self.model_runner.moe_layers
        self.moe_fusions = self.model_runner.moe_fusions

        # --- backend ---------------------------------------------------
        # Backends needing stable addresses for captured prefill segments
        # (today: only Breakable) allocate their own static buffers in
        # ``setup_prefill_state``. Other backends no-op.
        self.backend = resolve_prefill_backend(model_runner)
        self.backend.setup_prefill_state(self)
        self.backend.prepare(self)

        # --- capture --------------------------------------------------
        self.device_module.synchronize()
        self.model_runner.tp_group.barrier()
        self.capture()

        self.raw_num_tokens = 0

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _is_mamba_track_enabled(self) -> bool:
        return (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and not self.model_runner.server_args.disable_radix_cache
            and self.model_runner.spec_algorithm.is_none()
        )

    def _cache_loc_dtype(self):
        return torch.int64 if not is_npu() else torch.int32

    def _build_capture_forward_batch(self, num_tokens: int) -> ForwardBatch:
        """Build a dummy prefill ForwardBatch for capture/warmup at this shape.

        Default tensor inputs are fresh literals; backends that need
        stable addresses for captured segments (Breakable) override via
        ``populate_prefill_dummy_inputs`` to swap in static buffers.
        """
        buffers = self.buffers
        bs = 1

        with torch.device(self.device):
            shape_inputs = {
                "req_pool_indices": torch.arange(bs, device=self.device),
                "seq_lens": torch.tensor([num_tokens], device=self.device),
                "orig_seq_lens": torch.tensor([num_tokens], device=self.device),
                "extend_seq_lens": torch.tensor([num_tokens], device=self.device),
                "extend_prefix_lens": torch.tensor([0], device=self.device),
                "extend_start_loc": torch.tensor([0], device=self.device),
            }
        self.backend.populate_prefill_dummy_inputs(
            shape_inputs, bs=bs, num_tokens=num_tokens
        )

        with torch.device(self.device):
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=bs,
                input_ids=buffers.input_ids[:num_tokens],
                input_embeds=(
                    buffers.input_embeds[:num_tokens] if self.is_multimodal else None
                ),
                req_pool_indices=shape_inputs["req_pool_indices"],
                seq_lens=shape_inputs["seq_lens"],
                next_token_logits_buffer=None,
                orig_seq_lens=shape_inputs["orig_seq_lens"],
                seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                out_cache_loc=buffers.out_cache_loc[:num_tokens],
                out_cache_loc_swa=(
                    buffers.out_cache_loc_swa[:num_tokens]
                    if buffers.out_cache_loc_swa is not None
                    else None
                ),
                seq_lens_sum=num_tokens,
                mamba_track_indices=(
                    buffers.mamba_track_indices[:bs]
                    if buffers.mamba_track_indices is not None
                    else None
                ),
                mamba_track_mask=(
                    buffers.mamba_track_mask[:bs]
                    if buffers.mamba_track_mask is not None
                    else None
                ),
                mamba_track_seqlens=(
                    buffers.mamba_track_seqlens[:bs]
                    if buffers.mamba_track_seqlens is not None
                    else None
                ),
                encoder_lens=None,
                return_logprob=False,
                extend_num_tokens=num_tokens,
                extend_seq_lens=shape_inputs["extend_seq_lens"],
                extend_prefix_lens=shape_inputs["extend_prefix_lens"],
                extend_start_loc=shape_inputs["extend_start_loc"],
                extend_prefix_lens_cpu=torch.tensor([0], device="cpu"),
                extend_seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                extend_logprob_start_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                positions=buffers.positions[:num_tokens],
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
                global_dp_buffer_len=None,
                mrope_positions=(
                    buffers.mrope_positions[:, :num_tokens]
                    if self.is_multimodal
                    else None
                ),
                spec_algorithm=None,
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_token_non_padded=None,
                num_token_non_padded_cpu=num_tokens,
                global_forward_mode=ForwardMode.EXTEND,
                lora_ids=None,
                return_pooled_hidden_states=self.capture_return_pooled_hidden_states,
            )
            self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)
        return forward_batch

    def _run_forward(self, forward_batch: ForwardBatch, num_tokens: int):
        """Run model.forward inside the prefill set_forward_context."""
        forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
        set_dp_buffer_len(None, num_tokens, forward_batch.dp_padding_mode.is_max_len())
        set_is_extend_in_batch(False)

        with set_forward_context(
            forward_batch,
            self.attention_layers,
            self.quant_config,
            self.moe_layers,
            self.moe_fusions,
        ):
            return self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

    def _run_dummy_forward(self, num_tokens: int) -> None:
        """Build a dummy ForwardBatch at this shape, init attn metadata,
        run forward once. Used by ``TcPiecewiseCudaGraphBackend.prepare``
        for both the JIT-activate forward (single shape, before
        torch.compile install) and the compile-loop pass (every shape,
        inside ``enable_torch_compile_warmup``).
        """
        fb = self._build_capture_forward_batch(num_tokens)
        self.model_runner.attn_backend.init_forward_metadata(fb)
        self._run_forward(fb, num_tokens)

    # -----------------------------------------------------------------
    # can_run
    # -----------------------------------------------------------------
    def can_run(self, forward_batch: ForwardBatch) -> bool:
        if forward_batch.input_embeds is not None:
            return False
        if forward_batch.replace_embeds is not None:
            return False
        # tc_piecewise captures with ForwardMode.EXTEND and spec_info=None.
        if forward_batch.forward_mode.is_target_verify():
            return False
        if forward_batch.capture_hidden_mode != self.capture_hidden_mode:
            return False
        num_tokens = len(forward_batch.input_ids)
        if forward_batch.return_logprob:
            for start_len, seq_len in zip(
                forward_batch.extend_logprob_start_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
            ):
                if start_len is not None and start_len < seq_len:
                    return False
        if num_tokens > self.max_num_tokens:
            return False
        # Backend-level checks (e.g. Breakable rejects bs>1 prefill).
        return self.backend.can_run(forward_batch)

    # -----------------------------------------------------------------
    # capture loop
    # -----------------------------------------------------------------
    def capture(self) -> None:
        with freeze_gc(
            self.model_runner.server_args.enable_cudagraph_gc
        ), graph_capture() as graph_capture_context:
            stream = graph_capture_context.stream
            with self.backend.capture_session(stream):
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                capture_range = (
                    tqdm.tqdm(list(reversed(self.capture_num_tokens)))
                    if get_tensor_model_parallel_rank() == 0
                    else reversed(self.capture_num_tokens)
                )
                for num_tokens in capture_range:
                    if get_tensor_model_parallel_rank() == 0:
                        avail_mem = get_available_gpu_memory(
                            self.model_runner.device,
                            self.model_runner.gpu_id,
                            empty_cache=False,
                        )
                        capture_range.set_description(
                            f"Capturing num tokens ({num_tokens=} {avail_mem=:.2f} GB)"
                        )
                    self.capture_one_shape(num_tokens)

    def capture_one_shape(self, size: int) -> None:
        """Per-shape capture: build dummy ForwardBatch + run_once,
        delegate to backend. ``size`` is the prefill token count.
        """
        num_tokens = size
        forward_batch = self._build_capture_forward_batch(num_tokens)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            return self._run_forward(forward_batch, num_tokens)

        self.backend.capture_one(num_tokens, run_once, dummies=None)

    # -----------------------------------------------------------------
    # replay_prepare
    # -----------------------------------------------------------------
    def replay_prepare(self, forward_batch: ForwardBatch, **kwargs) -> ForwardBatch:
        """Pad, populate static buffers, and build the static_forward_batch
        the model code reads during replay.
        """
        buffers = self.buffers
        num_tokens = len(forward_batch.input_ids)
        static_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
        self.raw_num_tokens = num_tokens

        bs = forward_batch.batch_size

        swa_translator = (
            self.model_runner.token_to_kv_pool_allocator.translate_loc_from_full_to_swa
            if buffers.out_cache_loc_swa is not None
            else None
        )
        buffers.populate_from_forward_batch(
            forward_batch=forward_batch,
            raw_num_tokens=num_tokens,
            static_num_tokens=static_num_tokens,
            is_multimodal=self.is_multimodal,
            swa_translator=swa_translator,
        )

        out_cache_loc_swa = (
            buffers.out_cache_loc_swa[:static_num_tokens]
            if buffers.out_cache_loc_swa is not None
            else None
        )

        mamba_track_indices = (
            buffers.mamba_track_indices[:bs]
            if buffers.mamba_track_indices is not None
            else None
        )
        mamba_track_mask = (
            buffers.mamba_track_mask[:bs]
            if buffers.mamba_track_mask is not None
            else None
        )
        mamba_track_seqlens = (
            buffers.mamba_track_seqlens[:bs]
            if buffers.mamba_track_seqlens is not None
            else None
        )

        input_ids = buffers.input_ids[:static_num_tokens]
        input_embeds = (
            buffers.input_embeds[:static_num_tokens] if self.is_multimodal else None
        )
        positions = buffers.positions[:static_num_tokens]
        out_cache_loc = buffers.out_cache_loc[:static_num_tokens]
        mrope_positions = (
            buffers.mrope_positions[:, :static_num_tokens]
            if forward_batch.mrope_positions is not None
            else None
        )

        # Normalize MIXED→EXTEND so dynamo's guard (captured with EXTEND=1)
        # doesn't fail on MIXED=3.
        pcg_forward_mode = (
            ForwardMode.EXTEND
            if forward_batch.forward_mode == ForwardMode.MIXED
            else forward_batch.forward_mode
        )
        pcg_global_forward_mode = (
            ForwardMode.EXTEND
            if forward_batch.global_forward_mode == ForwardMode.MIXED
            else forward_batch.global_forward_mode
        )

        static_forward_batch = ForwardBatch(
            forward_mode=pcg_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            input_embeds=input_embeds,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            next_token_logits_buffer=None,
            orig_seq_lens=forward_batch.orig_seq_lens,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            out_cache_loc_swa=out_cache_loc_swa,
            seq_lens_sum=forward_batch.seq_lens_sum,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=mamba_track_seqlens,
            encoder_lens=forward_batch.encoder_lens,
            return_logprob=False,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            extend_start_loc=forward_batch.extend_start_loc,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_num_tokens=forward_batch.extend_num_tokens,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            positions=positions,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=forward_batch.dp_padding_mode,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=forward_batch.spec_algorithm,
            spec_info=forward_batch.spec_info,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            num_token_non_padded=forward_batch.num_token_non_padded,
            num_token_non_padded_cpu=forward_batch.num_token_non_padded_cpu,
            global_forward_mode=pcg_global_forward_mode,
            lora_ids=forward_batch.lora_ids,
            sampling_info=forward_batch.sampling_info,
            mm_inputs=forward_batch.mm_inputs,
            temp_scaled_logprobs=forward_batch.temp_scaled_logprobs,
            temperature=forward_batch.temperature,
            top_p_normalized_logprobs=forward_batch.top_p_normalized_logprobs,
            top_p=forward_batch.top_p,
            dimensions=forward_batch.dimensions,
            return_pooled_hidden_states=(
                self.capture_return_pooled_hidden_states
                or forward_batch.return_pooled_hidden_states
            ),
        )

        if out_cache_loc_swa is not None:
            self.model_runner.token_to_kv_pool.set_swa_loc(out_cache_loc_swa)

        # Backends that need stable addresses for captured segments
        # (Breakable) commit serving-time values into their static buffers.
        # Other backends no-op.
        self.backend.commit_prefill_serving_inputs(forward_batch)

        self._static_num_tokens = static_num_tokens
        return static_forward_batch

    # -----------------------------------------------------------------
    # replay
    # -----------------------------------------------------------------
    def replay(
        self, forward_batch: ForwardBatch, **kwargs
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        with self.backend.runtime_session():
            static_forward_batch = self.replay_prepare(forward_batch, **kwargs)

            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
            with set_forward_context(
                static_forward_batch,
                self.attention_layers,
                self.quant_config,
                self.moe_layers,
                self.moe_fusions,
            ):
                output = self.backend.replay(
                    self._static_num_tokens, static_forward_batch, **kwargs
                )

            if isinstance(output, LogitsProcessorOutput):
                # Preserve mm_input_embeds for speculative decoding.
                mm_input_embeds = None
                if (
                    self.model_runner.spec_algorithm.is_speculative()
                    and output.mm_input_embeds is not None
                ):
                    mm_input_embeds = output.mm_input_embeds[: self.raw_num_tokens]
                return LogitsProcessorOutput(
                    next_token_logits=output.next_token_logits[: self.raw_num_tokens],
                    hidden_states=(
                        output.hidden_states[: self.raw_num_tokens]
                        if output.hidden_states is not None
                        else None
                    ),
                    mm_input_embeds=mm_input_embeds,
                )
            elif isinstance(output, EmbeddingPoolerOutput):
                return output
            else:
                assert isinstance(output, PPProxyTensors)
                raise NotImplementedError(
                    "PPProxyTensors is not supported in PrefillCudaGraphRunner yet."
                )
