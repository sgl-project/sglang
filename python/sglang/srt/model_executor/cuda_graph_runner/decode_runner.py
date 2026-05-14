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
"""DecodeCudaGraphRunner — runs DECODE / TARGET_VERIFY / DLLM_EXTEND under
a pluggable backend.

Backend selection comes from ``cuda_graph_mode[Phase.DECODE]``:
  - ``"full"``      — default, ``FullCudaGraphBackend``: one
                      ``torch.cuda.CUDAGraph`` per shape.
  - ``"breakable"`` — experimental, ``BreakableCudaGraphBackend``:
                      segmented capture (no torch.compile).
  - ``"tc_piecewise"``     — not implemented for decode; logs a one-shot warning
                      and falls back to ``"full"``.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import tqdm
from torch.profiler import ProfilerActivity, profile

from sglang.srt.compilation.torch_compile_decoration import (
    patch_model,
    set_torch_compile_config,
)
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import (
    graph_capture,
    set_pdmux_status,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import should_record_nolora_graph
from sglang.srt.model_executor.cuda_graph_backend.breakable_cudagraph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.factory import resolve_decode_backend
from sglang.srt.model_executor.cuda_graph_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
    BaseCudaGraphRunner,
    freeze_gc,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_executor.cuda_graph_runner_utils.buffers import (
    DecodeInputBuffers,
)
from sglang.srt.model_executor.cuda_graph_runner_utils.capture_mode import (
    _set_capture_lora_variant,
    model_capture_mode,
)
from sglang.srt.model_executor.cuda_graph_runner_utils.deepep_adapter import (
    DeepEPCudaGraphRunnerAdapter,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    compute_local_num_token_non_padded,
    enable_num_token_non_padded,
)
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx, get_stream_groups
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


def _make_graph_key(bs, stream_idx=None, variant_label=None):
    """Build a graph dict key from batch size, stream index, and lora variant.

    Standalone function so speculative runners (which don't subclass
    DecodeCudaGraphRunner) can use the same key encoding.
    """
    key = bs if stream_idx is None else f"{stream_idx}_{bs}"
    if variant_label is not None:
        key = f"{variant_label}_{key}"
    return key


class DecodeCudaGraphRunner(BaseCudaGraphRunner):
    """Decode-phase CUDA graph runner.

    Owns: static input buffers (``DecodeInputBuffers``), capture-bs list,
    attention backend, two-batch-overlap plugin, DeepEP adapter, and the
    pluggable ``self.backend`` that handles the actual capture/replay.
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        *,
        attn_backend=None,
        speculative_num_steps: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        super().__init__(model_runner)
        # --- core state ------------------------------------------------
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.use_ngram_embedding = model_runner.use_ngram_embedding
        if self.use_ngram_embedding:
            hf_config = model_runner.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = model_runner.server_args.enable_pdmux

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()
        self.record_nolora_graph = should_record_nolora_graph()

        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm = self.dllm_config is not None
        self.attn_backend = attn_backend or model_runner.attn_backend
        self.speculative_num_steps = (
            model_runner.server_args.speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
            if speculative_num_draft_tokens is None
            else speculative_num_draft_tokens
        )

        # --- capture mode + tokens-per-bs ------------------------------
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1
        if model_runner.spec_algorithm.is_speculative():
            if self.model_runner.is_draft_worker:
                if not self.model_runner.spec_algorithm.is_dflash():
                    raise RuntimeError("This should not happen")
            self.capture_forward_mode = ForwardMode.TARGET_VERIFY
            self.num_tokens_per_bs = self.speculative_num_draft_tokens
        elif self.is_dllm:
            self.capture_forward_mode = ForwardMode.DLLM_EXTEND
            self.num_tokens_per_bs = self.dllm_config.block_size

        # --- bucket sizes ---------------------------------------------
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(
            model_runner, self.num_tokens_per_bs
        )
        log_info_on_rank0(logger, f"Capture cuda graph bs {self.capture_bs}")
        if KTRANSFORMERS_AVAILABLE:
            KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)

        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

        self.maybe_init_pdmux()
        self.seq_len_fill_value = (
            self.attn_backend.get_cuda_graph_seq_len_fill_value()
            if self.dllm_config is None
            else self.dllm_config.block_size
        )

        # Non-zero encoder length ensures cross-attention kernels are captured in the graph.
        self.encoder_len_fill_value = (
            getattr(model_runner.model_config.hf_config, "max_source_positions", 0)
            if self.is_encoder_decoder
            else 0
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        if self.model_runner.server_args.enable_lora:
            self.model_runner.lora_manager.init_cuda_graph_batch_info(
                max_bs_in_cuda_graph=self.max_bs,
                num_tokens_per_bs=self.num_tokens_per_bs,
            )

        enable_mamba_track = (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and self.model_runner.spec_algorithm.is_none()
        )

        if self.require_gathered_buffer:
            assert self.require_mlp_tp_gather or self.require_attn_tp_gather

        # --- buffers ---------------------------------------------------
        self.buffers: DecodeInputBuffers = DecodeInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            hidden_size=self.model_runner.model_config.hidden_size,
            vocab_size=self.model_runner.model_config.vocab_size,
            dtype=self.model_runner.model_config.dtype,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            is_encoder_decoder=self.is_encoder_decoder,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            seq_len_fill_value=self.seq_len_fill_value,
            encoder_len_fill_value=self.encoder_len_fill_value,
            num_tokens_per_bs=self.num_tokens_per_bs,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
            ne_token_table=(
                model_runner.token_table if self.use_ngram_embedding else None
            ),
            is_hybrid_swa=model_runner.is_hybrid_swa,
        )
        self.buffers.share_buffers()

        # --- backend ---------------------------------------------------
        self.backend = resolve_decode_backend(model_runner)
        self.backend.prepare(self)

        # --- capture --------------------------------------------------
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n" f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def maybe_init_pdmux(self):
        if self.enable_pdmux:
            self.stream_groups = get_stream_groups()
            for attn_backend in self.model_runner.decode_attn_backend_group:
                attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

    def _cache_loc_dtype(self):
        return torch.int64

    def _make_graph_key(self, bs, stream_idx=None, variant_label=None):
        return _make_graph_key(bs, stream_idx, variant_label)

    def _resolve_lora_variant(self, forward_batch: ForwardBatch):
        if not getattr(self, "record_nolora_graph", False):
            return None
        if forward_batch.lora_ids is not None and any(
            uid is not None for uid in forward_batch.lora_ids
        ):
            return "lora"
        return "nolora"

    # -----------------------------------------------------------------
    # can_run
    # -----------------------------------------------------------------
    def can_run(self, forward_batch: ForwardBatch):
        if forward_batch.replace_embeds is not None:
            return False
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                or self.model_runner.spec_algorithm.is_dflash()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.batch_size

        variant_label = self._resolve_lora_variant(forward_batch)
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        graph_key = self._make_graph_key(cuda_graph_bs, stream_idx, variant_label)

        is_bs_supported = (
            self.backend.has_shape(graph_key)
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )
        is_tbo_supported = (
            forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
        )

        is_ngram_supported = (
            (
                forward_batch.batch_size * self.num_tokens_per_bs
                == forward_batch.input_ids.numel()
            )
            if self.model_runner.spec_algorithm.is_ngram()
            else True
        )

        return (
            is_bs_supported
            and is_encoder_lens_supported
            and is_tbo_supported
            and capture_hidden_mode_matches
            and is_ngram_supported
            and self.backend.can_run(forward_batch)
        )

    # -----------------------------------------------------------------
    # Profiling helpers
    # -----------------------------------------------------------------
    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
        torch.cuda.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.cuda.memory._dump_snapshot("cuda_graph_runner_memory_usage.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by CUDA Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=10
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=10
            )
            + "\n\nMemory Usage is saved to cuda_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)

    # -----------------------------------------------------------------
    # capture loop
    # -----------------------------------------------------------------
    def capture(self) -> None:
        profile_context = empty_context()
        if self.enable_profile_cuda_graph:
            profile_context = self._init_profile_context_and_memory_record()

        def _capture_one_stream(stream_idx: Optional[int] = None):
            avail_mem = get_available_gpu_memory(
                self.model_runner.device,
                self.model_runner.gpu_id,
                empty_cache=False,
            )
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_bs)
            )
            lora_variants = (
                [("lora", True), ("nolora", False)]
                if getattr(self, "record_nolora_graph", False)
                else [(None, None)]
            )
            for i, bs in enumerate(capture_range):
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                    )

                for variant_label, _variant_has_lora in lora_variants:
                    _set_capture_lora_variant(variant_label)
                    with patch_model(
                        self.model_runner.model,
                        bs in self.compile_bs,
                        num_tokens=bs * self.num_tokens_per_bs,
                        tp_group=self.model_runner.tp_group,
                    ) as forward:
                        self.capture_one_shape(bs, forward, stream_idx, variant_label)

        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
            if not self.enable_pdmux:
                with graph_capture() as graph_capture_context, profile_context as prof:
                    self.stream = graph_capture_context.stream
                    with self.backend.capture_session(self.stream):
                        _capture_one_stream()
            else:
                set_pdmux_status(False)
                for i, sg in enumerate(self.stream_groups):
                    with (
                        graph_capture(stream=sg[1]) as graph_capture_context,
                        profile_context as prof,
                    ):
                        self.stream = graph_capture_context.stream
                        with self.backend.capture_session(self.stream):
                            _capture_one_stream(i)

        _set_capture_lora_variant(None)

        if self.enable_profile_cuda_graph:
            self._post_process_after_profile(prof)

    # -----------------------------------------------------------------
    # capture_one_shape — per-shape capture (size = batch size for decode)
    # -----------------------------------------------------------------
    def capture_one_shape(
        self,
        size: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
    ):
        bs = size
        buffers: DecodeInputBuffers = self.buffers
        num_tokens = bs * self.num_tokens_per_bs

        # Sanity-check: --debug-cuda-graph requires breakable backend.
        if self.model_runner.server_args.debug_cuda_graph:
            assert isinstance(
                self.backend, BreakableCudaGraphBackend
            ), "Breakable CUDA graph is required for --debug-cuda-graph"

        # Graph inputs
        input_ids = buffers.input_ids[:num_tokens]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        if self.is_encoder_decoder:
            encoder_lens = buffers.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]

        buffers.num_token_non_padded[...] = num_tokens
        if (
            enable_num_token_non_padded()
            and self.require_gathered_buffer
            and not self.nsa_enable_prefill_cp
        ):
            local = compute_local_num_token_non_padded(
                global_num_token_non_padded=buffers.num_token_non_padded,
                num_tokens_per_dp=num_tokens,
            )
            buffers.num_token_non_padded.copy_(local)

        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.model_runner.server_args.enable_lora:
            lora_ids = [None] * bs
        else:
            lora_ids = None

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

        if stream_idx is None:
            attn_backend = self.attn_backend
        else:
            assert self.enable_pdmux
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=None,
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
            lora_ids=lora_ids,
        )

        forward_batch.hisparse_coordinator = self.model_runner.hisparse_coordinator
        if forward_batch.hisparse_coordinator is not None:
            forward_batch.hisparse_coordinator.num_real_reqs.fill_(bs)

        if buffers.ngram_embedding_info is not None:
            forward_batch.ngram_embedding_info = buffers.ngram_embedding_info.slice(bs)

        self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

        if lora_ids is not None:
            self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

        attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.pp_size > 1
                and "pp_proxy_tensors" in inspect.signature(forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if (
                self.model_runner.spec_algorithm.is_dflash()
                and self.model_runner.is_draft_worker
                and "input_embeds" in inspect.signature(forward).parameters
            ):
                kwargs["input_embeds"] = buffers.input_embeds[:num_tokens]

            return forward(
                input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        self.deepep_adapter.capture(is_extend_in_batch=False)

        # SWA: write before capture so set_kv_buffer's Python branch
        # (if self.swa_loc is not None) takes the fast path, recording
        # GPU ops that read this buffer instead of the slow per-layer
        # translate_loc_from_full_to_swa fallback.
        if self.buffers.out_cache_loc_swa is not None:
            self.model_runner.token_to_kv_pool.set_swa_loc(
                self.buffers.out_cache_loc_swa[:num_tokens]
            )

        shape_key = self._make_graph_key(bs, stream_idx, variant_label)
        self.backend.capture_one(shape_key, run_once, dummies=None)

    # -----------------------------------------------------------------
    # recapture
    # -----------------------------------------------------------------
    def recapture_if_needed(self, forward_batch: ForwardBatch):
        capture_hidden_mode_required_by_forward_batch = (
            forward_batch.capture_hidden_mode
        )
        capture_hidden_mode_required_by_spec_info = (
            getattr(forward_batch.spec_info, "capture_hidden_mode", None)
            or CaptureHiddenMode.NULL
        )
        capture_hidden_mode_required_for_returning_hidden_states = (
            CaptureHiddenMode.FULL
            if self.model_runner.server_args.enable_return_hidden_states
            else CaptureHiddenMode.NULL
        )

        required_capture_hidden_mode = max(
            capture_hidden_mode_required_by_forward_batch,
            capture_hidden_mode_required_by_spec_info,
            capture_hidden_mode_required_for_returning_hidden_states,
        )

        if self.capture_hidden_mode != required_capture_hidden_mode:
            self.capture_hidden_mode = required_capture_hidden_mode
            self.backend.cleanup()
            self.capture()

    # -----------------------------------------------------------------
    # replay_prepare
    # -----------------------------------------------------------------
    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        buffers = self.buffers
        self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens / self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                or self.model_runner.spec_algorithm.is_dflash()
                else max_num_tokens
            )
            bs = self._pad_to_bucket(int(max_batch_size), self.capture_bs)
        else:
            bs = self._pad_to_bucket(raw_bs, self.capture_bs)

        buffers.populate_from_forward_batch(
            forward_batch=forward_batch,
            raw_bs=raw_bs,
            raw_num_token=raw_num_token,
            bs=bs,
            seq_len_fill_value=self.seq_len_fill_value,
            require_gathered_buffer=self.require_gathered_buffer,
            num_tokens_per_bs=self.num_tokens_per_bs,
            nsa_enable_prefill_cp=self.nsa_enable_prefill_cp,
            enable_num_token_non_padded_flag=enable_num_token_non_padded(),
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if (
            self.model_runner.spec_algorithm.is_dflash()
            and self.model_runner.is_draft_worker
            and forward_batch.input_embeds is not None
        ):
            buffers.input_embeds[:raw_num_token].copy_(forward_batch.input_embeds)
        if self.enable_two_batch_overlap:
            self.tbo_plugin.replay_prepare(
                forward_mode=self.capture_forward_mode,
                bs=bs,
                num_token_non_padded=len(forward_batch.input_ids),
                spec_info=forward_batch.spec_info,
            )
        if forward_batch.forward_mode.is_idle() and forward_batch.spec_info is not None:
            forward_batch.spec_info.custom_mask = buffers.custom_mask
        if self.enable_pdmux:
            stream_idx = get_current_stream_idx()
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]
        else:
            attn_backend = self.attn_backend
        # FIXME: implicit channel for backends (dsv4) that need forward_batch
        # in replay metadata prep. Should become a real param on the interface.
        attn_backend._replay_forward_batch = forward_batch
        attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            buffers.req_pool_indices[:bs],
            buffers.seq_lens[:bs],
            forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
            buffers.encoder_lens[:bs] if self.is_encoder_decoder else None,
            self.capture_forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=buffers.seq_lens_cpu[:bs],
        )
        attn_backend._replay_forward_batch = None

        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

        if self.model_runner.hisparse_coordinator is not None:
            self.model_runner.hisparse_coordinator.num_real_reqs.fill_(raw_bs)

    # -----------------------------------------------------------------
    # replay
    # -----------------------------------------------------------------
    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        self.deepep_adapter.replay()

        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
            if (
                self.model_runner.spec_algorithm.is_dflash()
                and self.model_runner.is_draft_worker
                and forward_batch.input_embeds is not None
            ):
                self.buffers.input_embeds[: self.raw_num_token].copy_(
                    forward_batch.input_embeds
                )

        variant_label = self._resolve_lora_variant(forward_batch)
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        graph_key = self._make_graph_key(self.bs, stream_idx, variant_label)

        # Backend dispatches by shape — ``static_forward_batch`` is the
        # forward_batch the model code reads from during replay; for
        # Full/Breakable backends it's unused (replay against static
        # buffers in place); for tc_piecewise-decode (not yet implemented) it
        # would feed args to the compiled callable.
        timer_ctx = (
            self.model_runner.device_timer.wrap(
                metadata={
                    "category": forward_batch.forward_mode.name.lower(),
                }
            )
            if self.model_runner.device_timer
            else contextlib.nullcontext()
        )
        with timer_ctx, self.backend.runtime_session():
            output = self.backend.replay(graph_key, forward_batch)

        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = (
                    output.full_logits[: self.raw_num_token]
                    if output.full_logits is not None
                    else None
                )
            else:
                full_logits = None
                next_token_logits = (
                    output.next_token_logits[: self.raw_num_token]
                    if output.next_token_logits is not None
                    else None
                )

            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
                customized_info=output.customized_info,
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})

    # -----------------------------------------------------------------
    # spec info
    # -----------------------------------------------------------------
    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if (
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:

                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.model_runner.spec_algorithm.is_standalone()
                    else CaptureHiddenMode.FULL
                )
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.buffers.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_cum_len=None,
                    spec_steps=self.speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=self.speculative_num_draft_tokens,
                    capture_hidden_mode=capture_mode,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )
        elif self.model_runner.spec_algorithm.is_dflash():
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput
            from sglang.srt.speculative.dflash_utils import (
                resolve_dflash_verify_mask_policy,
            )

            _, build_custom_mask = resolve_dflash_verify_mask_policy(
                self.model_runner.attn_backend
            )
            spec_info = DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                custom_mask=(
                    None
                    if (self.model_runner.is_draft_worker or not build_custom_mask)
                    else self.buffers.custom_mask
                ),
                capture_hidden_mode=(
                    CaptureHiddenMode.NULL
                    if self.model_runner.is_draft_worker
                    else CaptureHiddenMode.FULL
                ),
            )

        elif self.model_runner.spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_info import NgramVerifyInput

            spec_info = NgramVerifyInput(
                draft_token=None,
                tree_mask=self.buffers.custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                draft_token_num=self.num_tokens_per_bs,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        return spec_info
