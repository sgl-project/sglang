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

Backend selection comes from cuda_graph_config.decode:
  - "full"      — default, FullCudaGraphBackend: one
                      torch.cuda.CUDAGraph per shape.
  - "breakable" — experimental, BreakableCudaGraphBackend:
                      segmented capture (no torch.compile).
  - "tc_piecewise"     — not implemented for decode; logs a one-shot warning
                      and falls back to "full".
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import tqdm
from torch.profiler import ProfilerActivity, profile

from sglang.srt.compilation import torch_compile_decoration
from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.distributed.parallel_state import (
    graph_capture,
    set_pdmux_status,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    build_decode_registry,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    compute_local_num_token_non_padded,
    enable_num_token_non_padded,
    get_required_capture_hidden_mode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    BaseCudaGraphRunner,
    freeze_gc,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    maybe_flashinfer_autotune_speculative_draft,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.utils import resolve_decode_backend
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.model_executor.runner_utils.buffers import (
    DecodeInputBuffers,
)
from sglang.srt.model_executor.runner_utils.capture_mode import (
    _set_capture_dsa_variant,
    _set_capture_lora_variant,
    model_capture_mode,
)
from sglang.srt.model_executor.runner_utils.deepep_adapter import (
    DeepEPCudaGraphRunnerAdapter,
)
from sglang.srt.model_executor.runner_utils.shared_read_event import make_external_event
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx, get_stream_groups
from sglang.srt.runtime_context import (
    get_exec,
    get_flags,
    get_parallel,
    get_spec,
)
from sglang.srt.speculative.ragged_verify import resolve_ragged_verify_layout
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    is_hip,
    require_attn_tp_gather,
    require_mlp_tp_gather,
)
from sglang.srt.utils.device_timer import device_timer_ctx
from sglang.srt.utils.profile_utils import (
    export_cuda_graph_capture_trace,
    graph_capture_profile_dir,
)

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def ragged_verify_compact_graphs_enabled(spec_algorithm: SpeculativeAlgorithm) -> bool:
    if not spec_algorithm.supports_ragged_verify():
        return False
    from sglang.srt.speculative.ragged_verify import ragged_verify_compact_enabled

    return ragged_verify_compact_enabled()


def build_replay_fb_view(
    forward_batch: ForwardBatch,
    buffers: DecodeInputBuffers,
    bs: int,
    raw_bs: int,
    num_tokens: int,
    seq_len_fill_value: int,
    capture_forward_mode: ForwardMode,
    is_encoder_decoder: bool,
) -> SimpleNamespace:
    """Construct a ForwardBatch-like view for backend replay-side init.

    Combines the original forward_batch (for unpadded / per-iter
    fields like spec_info, out_cache_loc, and the runtime
    actual_forward_mode) with the padded capture-time buffers from
    buffers (for req_pool_indices, seq_lens, seq_lens_cpu,
    positions, encoder_lens).

    forward_mode is the capture-time mode (used by backends for
    bucket / dispatch decisions); actual_forward_mode is the
    runtime mode (may be IDLE while the captured graph targets DECODE
    — DSV4's replay metadata prep uses this for IDLE substitution).

    Subsumes the _replay_forward_batch side channel that DSV4 used to
    read out-of-band before the init_forward_metadata 3-method ABC.
    """
    return SimpleNamespace(
        batch_size=bs,
        forward_mode=capture_forward_mode,
        actual_forward_mode=forward_batch.forward_mode,
        input_ids=buffers.input_ids[:num_tokens],
        positions=buffers.positions[:num_tokens],
        req_pool_indices=buffers.req_pool_indices[:bs],
        seq_lens=buffers.seq_lens[:bs],
        seq_lens_sum=(
            None
            if forward_batch.seq_lens_sum is None
            else forward_batch.seq_lens_sum + (bs - raw_bs) * seq_len_fill_value
        ),
        # Propagate mirror absence: the pinned buffer is not refreshed when the
        # batch has no CPU mirror; a stale non-None tensor defeats None-guards.
        seq_lens_cpu=(
            None if forward_batch.seq_lens_cpu is None else buffers.seq_lens_cpu[:bs]
        ),
        num_padding=bs - raw_bs,
        encoder_lens=buffers.encoder_lens[:bs] if is_encoder_decoder else None,
        out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
        out_cache_loc_dsv4=getattr(forward_batch, "out_cache_loc_dsv4", None),
        # The mamba-track registry slot (VIRTUAL ids) is the v2p translate SOURCE
        # for the backend, which copies the result into its own static buffer and
        # reads THAT in the decode track-save — this slot is never mutated. None
        # when mamba-track is disabled, slice to [:bs] like every other buffer
        mamba_track_indices=(
            None
            if buffers.mamba_track_indices is None
            else buffers.mamba_track_indices[:bs]
        ),
        spec_info=forward_batch.spec_info,
    )


class DecodeCudaGraphRunner(BaseCudaGraphRunner):
    """Decode-phase CUDA graph runner.

    Owns: static input buffers (DecodeInputBuffers), capture-bs list,
    attention backend, two-batch-overlap plugin, DeepEP adapter, and the
    pluggable self.backend that handles the actual capture/replay.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        attn_backend=None,
        speculative_num_steps: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        super().__init__(model_runner)

        # In-graph metadata prep: shared buffers -> in-graph private data
        self.in_graph_metadata_prep_done: Optional[torch.cuda.Event] = None

        # --- core state ------------------------------------------------
        self.enable_torch_compile = get_flags().capture.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_mlp_tp_gather = require_mlp_tp_gather(
            model_runner.server_args
        ) and not self._forward_is_dp_local(model_runner)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        # Composite predicates derive from the instance values so the dp-local
        # draft exemption above stays consistent (require_gathered_buffer ==
        # mlp_tp_gather or attn_tp_gather; require_mlp_sync adds dp attention).
        self.require_gathered_buffer = (
            self.require_mlp_tp_gather or self.require_attn_tp_gather
        )
        self.require_mlp_sync = (
            get_parallel().config.enable_dp_attention or self.require_gathered_buffer
        )
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.use_ngram_embedding = model_runner.ngram_embedding_manager.enabled
        if self.use_ngram_embedding:
            hf_config = model_runner.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k
        self.speculative_algorithm = get_spec().speculative_algorithm
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )

        # --- DSA dense-decode dual-graph -------------------------------
        # Capture a "dense" (k-only, skip-indexer) and a "sparse" (full indexer)
        # decode graph per bs bucket, and dispatch on max_kv_len vs index_topk at
        # replay. Auto-enabled for DSA models (index_topk present in the HF
        # config) — correct for mixed lengths since any request with
        # kv_len > index_topk falls back to the sparse graph. Adds ~52 graphs and
        # ~2x capture time.
        #
        # Scoped to HIP (AMD): the k-only dense-decode fast path has only been
        # validated on MI355X. This is common (non-hardware-gated) code, so on
        # CUDA we deliberately keep the original behavior (no dual-graph) to
        # avoid silently changing the CUDA decode path for DSA models (e.g.
        # DeepSeek-V3.2). CUDA can opt in later once validated there.
        self.dsa_dual_graph = False
        self.dsa_index_topk: Optional[int] = None
        from sglang.srt.configs.model_config import (
            get_dsa_index_topk,
            is_deepseek_dsa,
        )

        hf_config = model_runner.model_config.hf_config
        if is_hip() and is_deepseek_dsa(hf_config):
            self.dsa_index_topk = get_dsa_index_topk(hf_config)
            self.dsa_dual_graph = True
            logger.info(
                "[dense-decode] DSA dual-graph enabled: capturing "
                "dense (k-only) + sparse (full indexer) decode graphs; "
                "dispatch on max_kv_len vs index_topk=%d.",
                self.dsa_index_topk,
            )

        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        # True if a DSACPLayerCommunicator-style prefill-CP flavor is active
        # (DSA or MLA). These flavors feed a zigzag-split rank-local layout
        # into the runner; MHA-arch prefill CP (Qwen3/Qwen2 MoE via PR
        # #18233) uses the plain LayerCommunicator with an attn_tp-replicated
        # layout and is intentionally excluded so the attn_tp-local
        # num_token_non_padded adjustment still runs for it.
        self.enable_prefill_cp = (
            is_dsa_enable_prefill_cp() or is_mla_prefill_cp_enabled()
        )

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm = self.dllm_config is not None
        self.attn_backend = attn_backend or model_runner.attn_backend
        self.speculative_num_steps = (
            get_spec().speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        self.speculative_num_draft_tokens = (
            get_spec().speculative_num_draft_tokens
            if speculative_num_draft_tokens is None
            else speculative_num_draft_tokens
        )

        # --- capture mode + tokens-per-bs ------------------------------
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = self.return_hidden_states_mode
        # Static capture width.
        self.captured_req_width = model_runner.decode_num_tokens_per_req(
            num_draft_tokens=self.speculative_num_draft_tokens
        )
        if model_runner.spec_algorithm.is_speculative():
            if self.model_runner.is_draft_worker:
                # Draft workers can use TARGET_VERIFY mode.
                if (
                    not self.model_runner.spec_algorithm.supports_target_verify_for_draft()
                ):
                    raise RuntimeError("This should not happen")
            self.capture_forward_mode = ForwardMode.TARGET_VERIFY
        elif self.is_dllm:
            self.capture_forward_mode = ForwardMode.DLLM_EXTEND

        # --- bucket sizes ---------------------------------------------
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(
            model_runner, self.captured_req_width
        )
        if KTRANSFORMERS_AVAILABLE:
            KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)

        self.ragged_verify_mode = (
            ragged_verify_compact_graphs_enabled(self.model_runner.spec_algorithm)
            and (self.capture_forward_mode == ForwardMode.TARGET_VERIFY)
            and not self.model_runner.is_draft_worker
        )
        self.capture_num_tokens: Optional[list[int]] = (
            self._build_ragged_verify_token_buckets()
            if self.ragged_verify_mode
            else None
        )
        self._ragged_graph_size = 0
        # Per-tier capture layouts; their verify_lens / qo_indptr tensors are
        # baked into the captured graphs and refreshed in place each replay.
        self._captured_ragged_layouts: dict[int, object] = {}
        if self.ragged_verify_mode and (
            self.enable_two_batch_overlap
            or model_runner.lora_manager is not None
            or self.disable_padding
        ):
            raise ValueError(
                "Compact ragged verify does not support two-batch-overlap, "
                "LoRA, or disable-cuda-graph-padding (bs pads to the captured "
                "tier); disable SGLANG_RAGGED_VERIFY_MODE or the conflicting "
                "feature."
            )

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.captured_req_width
        self.attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

        # Init PDMux if needed
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

        if self.model_runner.lora_manager is not None:
            # Phase 2 of LoRA CUDA graph init: dense LoRA batch metadata.
            # Phase 1 (MoE buffers) was handled earlier in ModelRunner via
            # lora_manager.init_cuda_graph_moe_buffers().
            self.model_runner.lora_manager.init_cuda_graph_batch_info(
                max_bs_in_cuda_graph=self.max_bs,
                num_tokens_per_req=self.captured_req_width,
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
            next_token_logits_buffer=self.model_runner.graph_shared_output.get_logits_buffer(
                self.model_runner.model_config.vocab_size, rows=self.max_num_token
            ),
            dtype=self.model_runner.model_config.dtype,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            is_encoder_decoder=self.is_encoder_decoder,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            seq_len_fill_value=self.seq_len_fill_value,
            encoder_len_fill_value=self.encoder_len_fill_value,
            num_tokens_per_req=self.captured_req_width,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
            ne_token_table=(
                model_runner.ngram_embedding_manager.table
                if self.use_ngram_embedding
                else None
            ),
            hc_hidden_size=getattr(
                self.model_runner.model_config, "hc_hidden_size", None
            ),
            pp_proxy_topk_size=self.model_runner.get_pp_proxy_topk_size(),
            pp_proxy_residual_num_blocks=(
                self.model_runner.get_pp_proxy_residual_num_blocks()
            ),
        )
        self.buffers.share_buffers()
        # FB-shared slot registry adopting DecodeInputBuffers storage (same
        # physical tensors, stable data_ptr for capture vs replay). Provides
        # the unified fill_from / slot access surface, replacing
        # populate_from_forward_batch on capture/replay paths.
        self.buffer_registry: CudaGraphBufferRegistry = build_decode_registry(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            seq_len_fill_value=self.seq_len_fill_value,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
            is_encoder_decoder=self.is_encoder_decoder,
            encoder_len_fill_value=self.encoder_len_fill_value,
            enable_num_token_non_padded=enable_num_token_non_padded(),
            require_gathered_buffer=self.require_gathered_buffer,
            enable_prefill_cp=self.enable_prefill_cp,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            dp_size=self.dp_size,
            source=self.buffers,
        )

        # --- backend ---------------------------------------------------
        self.backend = resolve_decode_backend(self)

        # --- capture --------------------------------------------------
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n" f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _record_in_graph_metadata_prep_done(self):
        # Purely a marker at this point in the graph; where the shared reads
        # actually end is the attn backend's call.
        if not self.device_module.is_current_stream_capturing():
            # Warmup shares this body. Breakable capture still plants: it opens
            # segment 1 on context entry and every segment re-arms the node.
            # Routed through device_module so XPU (torch.xpu) is picked up
            # instead of hitting torch.cuda dummy stubs on non-CUDA builds.
            return
        if self.in_graph_metadata_prep_done is None:
            self.in_graph_metadata_prep_done = make_external_event(self.device_module)
        event = self.in_graph_metadata_prep_done
        if event is not None:
            # Stays None without external-event support, so the read-end
            # resolution below never hands out an unrecorded event.
            event.record()

    def _replay_attn_backend(self) -> AttentionBackend:
        # Under pdmux each stream replays on its own group member.
        if self.enable_pdmux:
            return self.model_runner.decode_attn_backend_group[get_current_stream_idx()]
        return self.attn_backend

    def _resolve_shared_read_ends(self, attn_backend, forward_mode) -> SharedReadEnds:
        declared = attn_backend.shared_read_ends(forward_mode)
        if (
            declared is SharedReadEnds.IN_REPLAY
            and self.in_graph_metadata_prep_done is None
        ):
            # TODO: this lands EARLIER than declared; POST_REPLAY is the sound one.
            return SharedReadEnds.PRE_REPLAY
        return declared

    def _publish_read_done(self, in_graph: bool):
        """Hand the scheduler's WAR barrier the event marking this phase's
        shared-buffer reads as done."""
        if in_graph:
            # Reads end at the in-graph marker: wire it through, don't re-record.
            self.model_runner.shared_read_done_event = self.in_graph_metadata_prep_done
        else:
            read_done = self.device_module.Event()
            read_done.record()
            self.model_runner.shared_read_done_event = read_done

    def _build_ragged_verify_token_buckets(self) -> list[int]:
        buckets = sorted({bs * self.captured_req_width for bs in self.capture_bs})
        assert buckets and buckets[0] > 0, f"{buckets=}"
        return buckets

    def _autotune_buffers(self):
        """Reuse these static decode buffers (sized to max_bs) for the warmup
        flashinfer-autotune dummy forward instead of allocating a throwaway set
        — see BaseRunner._autotune_buffers / BaseRunner._dummy_run.

        The dummy forward derives its shape from max_bs and must match these
        buffers exactly; _dummy_run asserts that. Every autotune-reachable
        decode shape (plain decode, spec target-verify) matches. DLLM would not
        (its buffers hold block_size tokens/bs while the dummy run derives 1),
        but DLLM does not use a flashinfer MoE backend, so autotune never runs
        for it and this is never reached there.
        """
        return self.buffers, self.max_bs

    def maybe_init_pdmux(self):
        if self.enable_pdmux:
            self.stream_groups = get_stream_groups()
            for attn_backend in self.model_runner.decode_attn_backend_group:
                attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

    def _cache_loc_dtype(self):
        return torch.int64

    def _make_graph_key(
        self, size, stream_idx=None, variant_label=None, dsa_variant=None
    ):
        return ShapeKey(
            size=size,
            stream_idx=stream_idx,
            variant_label=variant_label,
            dsa_variant=dsa_variant,
        )

    def _capture_graph_size(self, *, bs: int, num_tokens: int) -> int:
        return num_tokens if self.ragged_verify_mode else bs

    def _resolve_dsa_variant(self, forward_batch: ForwardBatch) -> Optional[str]:
        """Host dispatch: pick which pre-captured DSA decode graph to replay
        from the batch-max kv_len. If any request has kv_len > index_topk
        the dense (k-only) graph would be wrong for it, so the whole batch uses
        the sparse (full indexer) graph. Returns None when dual-graph is off."""
        if not getattr(self, "dsa_dual_graph", False):
            return None
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            # Host-side mirror (maintained incrementally for plain decode) — no
            # d2h sync needed.
            max_kv_len = int(seq_lens_cpu.max().item())
        elif forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0:
            # Fallback: a single scalar reduction d2h (cheap, per-step).
            max_kv_len = int(forward_batch.seq_lens.max().item())
        else:
            # No length info: be safe and use the correct-for-all sparse graph.
            return "sparse"
        return "dense" if max_kv_len <= self.dsa_index_topk else "sparse"

    def _resolve_lora_variant(self, forward_batch: ForwardBatch):
        if not getattr(self, "record_nolora_graph", False):
            return None
        if forward_batch.lora_ids is not None and any(
            uid is not None for uid in forward_batch.lora_ids
        ):
            return "lora"
        return "nolora"

    @staticmethod
    def _forward_is_dp_local(model_runner) -> bool:
        """The DSpark dense draft runs attn-TP-local (draft_tp_context): each
        DP rank drafts independently with no cross-DP collective, so its
        hand-built batches carry no dp-global metadata and must key graphs by
        local batch size. Everything else keeps the dp-global padding path."""
        if not model_runner.is_draft_worker:
            return False
        if not model_runner.spec_algorithm.is_dspark():
            return False
        from sglang.srt.speculative.dspark_components.dspark_config import (
            draft_is_deepseek_v4,
        )

        return not draft_is_deepseek_v4(server_args=model_runner.server_args)

    def _ragged_capture_slots(self, num_tokens: int) -> int:
        if envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.get():
            return num_tokens // self.captured_req_width
        return min(num_tokens, self.max_bs)

    def _capture_ragged_verify_layout(self, num_tokens: int):
        if not self.ragged_verify_mode:
            return None
        if envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.get():
            return None
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyLayout,
            build_capture_verify_lens,
        )

        verify_lens_cpu = build_capture_verify_lens(
            num_tokens=num_tokens,
            num_slots=self._ragged_capture_slots(num_tokens),
            num_draft_tokens=self.captured_req_width,
        )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=self.device,
            grid=self.capture_num_tokens,
        )
        self._captured_ragged_layouts[num_tokens] = layout
        return layout

    def _stage_ragged_verify_layout(self, ragged_layout, graph_size_key: int) -> None:
        # Without this refresh every replay reuses the capture-time synthetic
        # verify_lens / qo_indptr and mis-slices the packed q rows.
        cap_layout = self._captured_ragged_layouts.get(graph_size_key)
        if cap_layout is None:
            return
        live = ragged_layout
        if live.bs != cap_layout.bs or live.cap is None:
            live = live.padded_to_bucket(
                padded_bs=cap_layout.bs, cap=self.captured_req_width
            )
        cap_layout.verify_lens.copy_(live.verify_lens)
        cap_layout.qo_indptr_device.copy_(live.qo_indptr_device)

    @staticmethod
    def _max_dp_batch_size(forward_batch: ForwardBatch) -> int:
        request_counts = forward_batch.original_global_num_tokens_cpu
        if request_counts is None:
            raise RuntimeError(
                "DP CUDA graph replay requires raw per-rank request counts"
            )
        return max(request_counts)

    def can_run_graph(self, forward_batch: ForwardBatch):
        # Disable for token embedding overrides (dynamic per-request)
        if forward_batch.replace_embeds is not None:
            return False

        ragged_layout = (
            resolve_ragged_verify_layout(forward_batch)
            if self.ragged_verify_mode
            else None
        )
        if ragged_layout is not None:
            return self._can_run_ragged_verify_graph(forward_batch, ragged_layout)
        if self.ragged_verify_mode and forward_batch.forward_mode.is_target_verify():
            return False

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
            cuda_graph_bs = self._max_dp_batch_size(forward_batch)
        else:
            cuda_graph_bs = forward_batch.batch_size

        graph_key = self._make_graph_key(
            cuda_graph_bs,
            stream_idx=get_current_stream_idx() if self.enable_pdmux else None,
            variant_label=self._resolve_lora_variant(forward_batch),
        )

        is_bs_supported = (
            self.backend.can_run(forward_batch, graph_key)
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        is_tbo_supported = (
            forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
        )

        is_ngram_supported = (
            (
                forward_batch.batch_size * self.captured_req_width
                == forward_batch.input_ids.numel()
            )
            if self.model_runner.spec_algorithm.is_ngram()
            else True
        )

        return (
            is_bs_supported
            and is_encoder_lens_supported
            and is_tbo_supported
            and is_ngram_supported
        )

    def _can_run_ragged_verify_graph(self, forward_batch: ForwardBatch, ragged_layout):
        if not self.attn_backend.supports_ragged_verify_graph:
            return False

        admission_tokens = ragged_layout.graph_num_tokens
        is_tokens_supported = admission_tokens <= self.capture_num_tokens[
            -1
        ] and forward_batch.batch_size <= self._ragged_capture_slots(admission_tokens)

        is_dp_supported = (
            forward_batch.can_run_dp_cuda_graph if self.require_mlp_sync else True
        )

        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        capture_hidden_mode_matches = (
            forward_batch.capture_hidden_mode <= self.capture_hidden_mode
        )

        return (
            is_tokens_supported
            and is_dp_supported
            and is_encoder_lens_supported
            and capture_hidden_mode_matches
        )

    def _graph_batch_capture_active(self) -> bool:
        """Whether the per-batch-size capture-trace feature is active.

        Gated by SGLANG_GRAPH_BATCH_CAPTURE. The original single-trace export
        (SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE) takes precedence: when both are
        set we fall back to the original behavior.
        """
        return (
            envs.SGLANG_GRAPH_BATCH_CAPTURE.get()
            and not envs.SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE.get()
        )

    def _init_profile_context_and_memory_record(self):
        if self._graph_batch_capture_active():
            # Per-batch-size capture traces (SGLANG_GRAPH_BATCH_CAPTURE): a
            # scheduled profiler is stepped once per batch size (see
            # FullCudaGraphBackend.capture_one) and on_trace_ready writes one
            # chrome trace per bs.
            rank = get_parallel().tp_rank
            runner_name = type(self).__name__
            trace_dir = graph_capture_profile_dir()
            os.makedirs(trace_dir, exist_ok=True)

            # Track which BS is currently being captured for trace file naming
            self._profile_bs_list = list(reversed(self.capture_bs))
            self._profile_bs_idx = 0

            def on_trace_ready(prof):
                bs = self._profile_bs_list[self._profile_bs_idx]
                trace_file = os.path.join(
                    trace_dir, f"{runner_name}_bs_{bs}_rank{rank}.json.gz"
                )
                prof.export_chrome_trace(trace_file)
                logger.info(f"Saved trace for bs={bs} to {trace_file}")
                self._profile_bs_idx += 1

            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                # Schedule: wait=2 (skip 2 dummy runs), warmup=0, active=1
                # (capture run); repeat=0 repeats the cycle so each batch size
                # gets its own trace.
                schedule=torch.profiler.schedule(wait=2, warmup=0, active=1, repeat=0),
                record_shapes=True,
                with_stack=True,
                with_flops=True,
                profile_memory=True,
                on_trace_ready=on_trace_ready,
            )
        else:
            # a single unscheduled pass over the whole
            # capture. The combined trace (if any) is exported in
            # _post_process_after_profile via export_cuda_graph_capture_trace,
            # gated by SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE.
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

        # single-trace export for the whole capture pass; no-op unless
        # SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE is set. In per-bs mode
        # (SGLANG_GRAPH_BATCH_CAPTURE) that env is unset, so this stays a no-op
        # and the per-bs on_trace_ready handles export instead.
        export_cuda_graph_capture_trace(
            prof_context,
            runner_name=type(self).__name__,
            tp_rank=get_parallel().tp_rank,
        )

    def capture_prepare(
        self,
        size: int,
        stream_idx: Optional[int] = None,
        num_tokens: Optional[int] = None,
    ):
        """Build the dummy decode ForwardBatch for capture at size (=bs),
        populate static input buffers, choose the active attn backend, and
        optionally build pp_proxy_tensors.

        num_tokens defaults to the uniform bs * num_tokens_per_req; ragged
        verify capture passes the decoupled (slots, tier tokens) pair.

        Returns (forward_batch, attn_backend, pp_proxy_tensors);
        pp_proxy_tensors is None unless pp_size > 1.
        """
        bs = size
        buffers: DecodeInputBuffers = self.buffers
        if num_tokens is None:
            num_tokens = bs * self.captured_req_width

        # Registry-owned FB-shared slots come through the registry (which
        # shares physical storage with self.buffers via source=...); the rest
        # still come off buffers directly.
        registry = self.buffer_registry

        def _slot(name):
            return registry.get_slot(name).slice_for(bs, num_tokens)

        input_ids = _slot("input_ids")
        req_pool_indices = _slot("req_pool_indices")
        seq_lens = _slot("seq_lens")
        seq_lens_cpu = _slot("seq_lens_cpu")
        out_cache_loc = _slot("out_cache_loc")
        positions = _slot("positions")
        encoder_lens = (
            _slot("encoder_lens") if registry.has_slot("encoder_lens") else None
        )
        mrope_positions = _slot("mrope_positions")
        next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]
        rids_int = buffers.rids_int[:bs] if buffers.rids_int is not None else None
        bootstrap_room_ids_int = (
            buffers.bootstrap_room_ids_int[:bs]
            if buffers.bootstrap_room_ids_int is not None
            else None
        )

        # Adjust for attention TP if needed (matching replay path in
        # populate_from_forward_batch).
        buffers.num_token_non_padded[...] = num_tokens
        if (
            enable_num_token_non_padded()
            and self.require_gathered_buffer
            and not self.enable_prefill_cp
        ):
            local = compute_local_num_token_non_padded(
                global_num_token_non_padded=buffers.num_token_non_padded,
                num_tokens_per_dp=num_tokens,
            )
            buffers.num_token_non_padded.copy_(local)

        pp_proxy_tensors = None
        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather:
            global_num_tokens_cpu = [num_tokens] * self.dp_size
        elif self.require_attn_tp_gather:
            global_num_tokens_cpu = [num_tokens]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu, dtype=torch.int32, device=input_ids.device
            )
            buffers.global_num_tokens_gpu.copy_(num_tokens_tensor)
            buffers.global_num_tokens_for_logprob_gpu.copy_(num_tokens_tensor)
        else:
            global_dp_buffer_len = None

        spec_info = self.get_spec_info(num_tokens)
        self.capture_hidden_mode = get_required_capture_hidden_mode(
            self.capture_hidden_mode,
            spec_info,
        )

        if self.model_runner.lora_manager is not None:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        # mamba state tracking (registry-owned when enabled)
        mamba_track_indices = (
            _slot("mamba_track_indices")
            if registry.has_slot("mamba_track_indices")
            else None
        )
        mamba_track_mask = (
            _slot("mamba_track_mask") if registry.has_slot("mamba_track_mask") else None
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
            global_num_tokens_cpu=global_num_tokens_cpu,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
            lora_ids=lora_ids,
            rids_int=rids_int,
            bootstrap_room_ids_int=bootstrap_room_ids_int,
        )

        # Trip the coordinator so the hisparse code path is captured into the
        # graph; backends read it from self.model_runner.hisparse_coordinator.
        forward_batch.hisparse_coordinator = self.model_runner.hisparse_coordinator
        if forward_batch.hisparse_coordinator is not None:
            forward_batch.hisparse_coordinator.num_real_reqs.fill_(bs)

        if buffers.ngram_embedding_info is not None:
            forward_batch.ngram_embedding_info = buffers.ngram_embedding_info.slice(bs)

        return forward_batch, attn_backend, pp_proxy_tensors

    def capture(self) -> None:
        # Warm up + autotune kernels once before capture (run-once across the
        # decode + prefill runners; see BaseRunner.warmup).
        self.warmup()
        # warmup() may disable torch.compile for a model whose _can_torch_compile
        # is False; recompute the compile bucket so capture matches.
        if self.enable_torch_compile and not (get_flags().capture.enable_torch_compile):
            self.enable_torch_compile = False
            _, self.compile_bs = get_batch_sizes_to_capture(
                self.model_runner, self.captured_req_width
            )
        profile_context = empty_context()
        # Holds the active torch profiler during capture so the backend can
        # advance its schedule (profiler.step()) per batch size. Only the
        # scheduled per-bs profiler (SGLANG_GRAPH_BATCH_CAPTURE) needs stepping;
        # the original unscheduled pass leaves this None.
        self._profiler = None
        if self.enable_profile_cuda_graph:
            profile_context = self._init_profile_context_and_memory_record()
            if self._graph_batch_capture_active():
                self._profiler = profile_context

        # share_buffers() coalesces seq_lens / seq_lens_cpu through the process-
        # wide pool, so they may alias a buffer seeded by an earlier runner (the
        # eager registry fills them with 0). The capture-time attention-metadata
        # plan reads these as the per-request KV length, and the prefill wrapper
        # (DLLM_EXTEND) asserts kv_len >= qo_len, so restore the fill value the
        # captured graph needs before capturing.
        self.buffers.seq_lens.fill_(self.seq_len_fill_value)
        self.buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
        # Capture runs real forwards, so a mid-serving recapture would index --
        # and write KV -- through the previous batch's live values. Replay is
        # already covered by the registry's padding policy.
        self.buffers.reset_index_buffers()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
            if not self.enable_pdmux:
                with graph_capture() as graph_capture_context, profile_context as prof:
                    self.stream = graph_capture_context.stream
                    with self.backend.capture_session(self.stream):
                        self._capture_one_stream()
            else:
                set_pdmux_status(False)
                for i, sg in enumerate(self.stream_groups):
                    with (
                        graph_capture(stream=sg[1]) as graph_capture_context,
                        profile_context as prof,
                    ):
                        self.stream = graph_capture_context.stream
                        with self.backend.capture_session(self.stream):
                            self._capture_one_stream(i)

        if self.enable_profile_cuda_graph:
            self._post_process_after_profile(prof)
        self._profiler = None

        # No pool-side pin to clear: the captured full-physical write loc rides the
        # backend's `ForwardMetadata.out_cache_loc_full_physical` (-> KVWriteLoc.full_loc).

    def _capture_one_stream(self, stream_idx: Optional[int] = None) -> None:
        avail_mem = get_available_gpu_memory(
            self.model_runner.device,
            self.model_runner.gpu_id,
            empty_cache=False,
        )
        # Reverse so cuda graphs share memory better.
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_bs)))
            if get_parallel().tp_rank == 0
            else reversed(self.capture_bs)
        )
        lora_variants = (
            [("lora", True), ("nolora", False)]
            if getattr(self, "record_nolora_graph", False)
            else [(None, None)]
        )
        # DSA: capture a dense (k-only) and a sparse (full indexer) graph
        # per bs bucket. Order: dense first so its (smaller) capture-time peak
        # runs while the shared pool is fresh; sparse's peak subsumes it.
        # getattr default: subclasses like EAGLEDraftCudaGraphRunner reuse this
        # capture() but don't run DecodeCudaGraphRunner.__init__ (so they never
        # set dsa_dual_graph) and override capture_one_shape with a signature that
        # has no dsa_variant. Default to no dual-graph and, for the None variant,
        # call capture_one_shape without the extra arg so those overrides work.
        dsa_variants = (
            ["dense", "sparse"] if getattr(self, "dsa_dual_graph", False) else [None]
        )
        for bs in capture_range:
            if get_parallel().tp_rank == 0:
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
                for dsa_variant in dsa_variants:
                    _set_capture_dsa_variant(dsa_variant)
                    with torch_compile_decoration.patch_model(
                        self.model_runner.model,
                        bs in self.compile_bs,
                        num_tokens=bs * self.captured_req_width,
                        tp_group=self.model_runner.tp_group,
                    ) as forward:
                        if dsa_variant is None:
                            self.capture_one_shape(
                                bs, forward, stream_idx, variant_label
                            )
                        else:
                            self.capture_one_shape(
                                bs, forward, stream_idx, variant_label, dsa_variant
                            )
        _set_capture_dsa_variant(None)

    def capture_one_shape(
        self,
        size: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
        dsa_variant: Optional[str] = None,
    ):
        num_tokens = size * self.captured_req_width
        bs = self._ragged_capture_slots(num_tokens) if self.ragged_verify_mode else size

        # Sanity-check: --debug-cuda-graph requires breakable backend.
        if get_exec().graph.debug_cuda_graph:
            assert isinstance(
                self.backend, BreakableCudaGraphBackend
            ), "Breakable CUDA graph is required for --debug-cuda-graph"

        forward_batch, attn_backend, pp_proxy_tensors = self.capture_prepare(
            bs, stream_idx=stream_idx, num_tokens=num_tokens
        )

        # All setup hooks below read get_attn_backend() (TboForwardBatchPreparer,
        # DeepEP adapter, …) so they must run inside the same ForwardContext
        # that wraps the warmup/capture forward.
        with forward_context(ForwardContext(attn_backend=attn_backend)):
            self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

            if forward_batch.lora_ids is not None:
                self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

            attn_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)

            def run_once():
                # Graph-recordable metadata-prep hook. The unified memory pool
                # records ZERO translate nodes here: all its read/write translates
                # run eagerly in `init_forward_metadata_out_graph` (replay-prep), so
                # the captured graph reads already-physical locs. Base no-op for triton.
                attn_backend.init_forward_metadata_in_graph(forward_batch)
                self._record_in_graph_metadata_prep_done()

                # No invalidate_loc_cache() here: the unified pool translates its
                # locs in `init_forward_metadata_out_graph`, so no cache to invalidate.

                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                    None
                )
                set_dp_buffer_len(
                    forward_batch.global_dp_buffer_len,
                    num_tokens,
                    forward_batch.dp_padding_mode.is_max_len(),
                    forward_batch.global_num_tokens_cpu,
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
                    self.model_runner.spec_algorithm.is_dflash_family()
                    and self.model_runner.is_draft_worker
                    and "input_embeds" in inspect.signature(forward).parameters
                    and not hasattr(self.model_runner.model, "forward_embed")
                ):
                    kwargs["input_embeds"] = self.buffers.input_embeds[:num_tokens]

                out = forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
                for capture_hook in self.model_runner.capture_tail_hooks:
                    capture_hook(self, out, forward_batch, num_tokens)
                return out

            self.deepep_adapter.capture(is_extend_in_batch=False)
            canary_ctx = (
                c.with_active_single_forward_manager(0)
                if (c := self.model_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            # Full-physical write loc lives in the attention metadata (the backend's
            # `out_cache_loc_full_physical` -> KVWriteLoc.full_loc), so the runner
            # wires no buffer here. (SWA write loc rides the `swa_out_cache_loc` rail.)

            with canary_ctx:
                shape_key = self._make_graph_key(
                    self._capture_graph_size(bs=bs, num_tokens=num_tokens),
                    stream_idx,
                    variant_label,
                    dsa_variant,
                )
                post_warmup_hook = getattr(
                    self.model_runner.attn_backend,
                    "on_after_cuda_graph_warmup",
                    None,
                )
                maybe_flashinfer_autotune_speculative_draft(
                    self,
                    run_once,
                    post_warmup_hook=post_warmup_hook,
                    run_lm_head=True,
                )
                self.backend.capture_one(
                    shape_key,
                    run_once,
                    capture_inputs=None,
                    post_warmup_hook=post_warmup_hook,
                )

    def _validate_capture_hidden_mode(self, forward_batch: ForwardBatch) -> None:
        if self.capture_hidden_mode < forward_batch.capture_hidden_mode:
            raise RuntimeError(
                "The runtime hidden-state mode exceeds the fixed CUDA graph "
                f"capture mode ({self.capture_hidden_mode.name})."
            )

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        ragged_layout = (
            resolve_ragged_verify_layout(forward_batch)
            if self.ragged_verify_mode
            else None
        )
        is_ragged = ragged_layout is not None

        self.deepep_adapter.replay()

        if not forward_batch.needs_forward_metadata_init():
            # Pre-planned (plan-stream load_batch already ran).
            # In speculative decoding, these two fields are still needed.
            graph_size_key = (
                self._ragged_graph_size
                if is_ragged
                else self._capture_graph_size(
                    bs=self.bs, num_tokens=self.bs * self.captured_req_width
                )
            )
            if is_ragged:
                assert self.raw_num_token == ragged_layout.graph_num_tokens, (
                    f"stale ragged raw_num_token {self.raw_num_token} != "
                    f"{ragged_layout.graph_num_tokens}"
                )
                self._stage_ragged_verify_layout(ragged_layout, graph_size_key)
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
            if (
                not is_ragged
                and self.model_runner.spec_algorithm.is_dflash_family()
                and self.model_runner.is_draft_worker
                and forward_batch.input_embeds is not None
            ):
                self.buffers.input_embeds[: self.raw_num_token].copy_(
                    forward_batch.input_embeds
                )
            variant_label = self._resolve_lora_variant(forward_batch)
            dsa_variant = self._resolve_dsa_variant(forward_batch)
            stream_idx = get_current_stream_idx() if self.enable_pdmux else None
            self._replay_graph_key = self._make_graph_key(
                graph_size_key, stream_idx, variant_label, dsa_variant
            )
            return

        buffers = self.buffers
        self._validate_capture_hidden_mode(forward_batch)

        raw_bs = forward_batch.batch_size

        if is_ragged:
            raw_num_token = ragged_layout.graph_num_tokens
            graph_size_key = self._ragged_graph_num_tokens(raw_num_token)
            assert graph_size_key == ragged_layout.graph_num_tokens, (
                f"ragged verify tier mismatch: runner tier {graph_size_key} != "
                f"layout graph_num_tokens {ragged_layout.graph_num_tokens}"
            )
            bs = self._ragged_capture_slots(graph_size_key)
            assert bs >= raw_bs, (
                f"ragged capture slots {bs} (tier {graph_size_key}) < raw_bs "
                f"{raw_bs}; the planner must reject this batch before replay"
            )
            padded_num_tokens = graph_size_key
            self._stage_ragged_verify_layout(ragged_layout, graph_size_key)
        else:
            raw_num_token = raw_bs * self.captured_req_width
            if self.require_mlp_tp_gather:
                max_batch_size = self._max_dp_batch_size(forward_batch)
                bs = self._pad_to_bucket(max_batch_size, self.capture_bs)
            else:
                bs = self._pad_to_bucket(raw_bs, self.capture_bs)
            padded_num_tokens = bs * self.captured_req_width
            graph_size_key = self._capture_graph_size(
                bs=bs, num_tokens=padded_num_tokens
            )

        self.buffer_registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=bs,
            raw_num_tokens=raw_num_token,
            padded_num_tokens=padded_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if (
            not is_ragged
            and self.model_runner.spec_algorithm.is_dflash_family()
            and self.model_runner.is_draft_worker
            and forward_batch.input_embeds is not None
        ):
            buffers.input_embeds[:raw_num_token].copy_(forward_batch.input_embeds)
        # Padded tokens aren't read, so skip zeroing. Ragged input_ids arrive
        # from the planner already padded to the tier, invalid slots zeroed.
        if self.enable_two_batch_overlap:
            self.tbo_plugin.replay_prepare(
                forward_mode=self.capture_forward_mode,
                bs=bs,
                num_token_non_padded=len(forward_batch.input_ids),
                spec_info=forward_batch.spec_info,
            )
        if (
            not is_ragged
            and forward_batch.forward_mode.is_idle()
            and forward_batch.spec_info is not None
        ):
            forward_batch.spec_info.custom_mask = buffers.custom_mask

        attn_backend = self._replay_attn_backend()
        fb_view = build_replay_fb_view(
            forward_batch=forward_batch,
            buffers=buffers,
            bs=bs,
            raw_bs=raw_bs,
            num_tokens=padded_num_tokens,
            seq_len_fill_value=self.seq_len_fill_value,
            capture_forward_mode=self.capture_forward_mode,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        attn_backend.init_forward_metadata_out_graph(fb_view)

        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs
        if is_ragged:
            self._ragged_graph_size = graph_size_key

        if self.model_runner.hisparse_coordinator is not None:
            self.model_runner.hisparse_coordinator.num_real_reqs.fill_(raw_bs)

        variant_label = self._resolve_lora_variant(forward_batch)
        dsa_variant = self._resolve_dsa_variant(forward_batch)
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        self._replay_graph_key = self._make_graph_key(
            graph_size_key, stream_idx, variant_label, dsa_variant
        )

    def _ragged_graph_num_tokens(self, total_verify_tokens: int) -> int:
        from sglang.srt.speculative.ragged_verify import round_up_grid

        return round_up_grid(total_verify_tokens, self.capture_num_tokens)

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        timer_ctx = device_timer_ctx(
            self.model_runner.device_timer, forward_batch.forward_mode.name.lower()
        )
        shared_read_ends = self._resolve_shared_read_ends(
            self._replay_attn_backend(), forward_batch.forward_mode
        )
        with timer_ctx, self.backend.replay_session():
            self.load_batch(forward_batch, pp_proxy_tensors)
            if envs.SGLANG_LOG_DECODE_GRAPH_KEY.get():
                logger.info(
                    "Decode graph replay: worker=%s key_size=%s (%s) mode=%s raw_bs=%d%s",
                    "draft" if self.model_runner.is_draft_worker else "target",
                    self._replay_graph_key.size,
                    "num_tokens" if self.ragged_verify_mode else "bs",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    (
                        f" slots={self._ragged_capture_slots(self._replay_graph_key.size)}"
                        if self.ragged_verify_mode
                        else ""
                    ),
                )
            if shared_read_ends is SharedReadEnds.PRE_REPLAY:
                self._publish_read_done(in_graph=False)

            output = self.backend.replay(self._replay_graph_key, forward_batch)

            if shared_read_ends is SharedReadEnds.IN_REPLAY:
                self._publish_read_done(in_graph=True)

            if shared_read_ends is SharedReadEnds.POST_REPLAY:
                self._publish_read_done(in_graph=False)

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
                    topk=get_spec().speculative_eagle_topk,
                    draft_token_num=self.speculative_num_draft_tokens,
                    capture_hidden_mode=capture_mode,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )
                # MTP models (e.g. deepseek_nextn) read spec_info.hidden_states
                spec_info.hidden_states = torch.zeros(
                    (num_tokens, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                    device=self.model_runner.device,
                )
        elif self.model_runner.spec_algorithm.is_dflash_family():
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput
            from sglang.srt.speculative.dflash_utils import (
                resolve_dflash_verify_mask_policy,
            )

            # Avoid enabling custom-mask modes during graph capture for backends that
            # can express DFLASH verify via their built-in causal path.
            _, build_custom_mask = resolve_dflash_verify_mask_policy(
                self.model_runner.attn_backend
            )
            spec_info = DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=self.captured_req_width,
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
                ragged_verify_layout=self._capture_ragged_verify_layout(num_tokens),
            )

        elif self.model_runner.spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_info import NgramVerifyInput

            spec_info = NgramVerifyInput(
                draft_token=None,
                custom_mask=self.buffers.custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                draft_token_num=self.captured_req_width,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        return spec_info
