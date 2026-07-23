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

Backend selection comes from cuda_graph_config.prefill:
  - "breakable"    — default on CUDA, BreakableCudaGraphBackend:
                      segmented capture (no torch.compile). Captures the
                      transformer body with one request slot, then replays it
                      with live batch metadata; multi-request prefill is
                      supported by running attention metadata and the
                      LM-head/logits tail outside the captured body.
  - "full"         — FullCudaGraphBackend: one graph per num_tokens bucket
                      for the captured transformer body. Capture uses
                      cuda_graph_config.prefill.full_prefill_max_req request
                      slots (auto-derived when unset); replay pads num_tokens
                      to the nearest bucket and pads unused request slots with
                      zero-length sentinels. bs > slots falls back to eager.
                      Attention metadata is refreshed out-of-graph against the
                      slot-padded batch before capture/replay.
  - "tc_piecewise" — TcPiecewiseCudaGraphBackend: torch.compile
                      wraps the model; per-shape compiled/captured pieces live
                      in torch.compile's internal cache. Multi-request prefill
                      is supported.
  - "disabled"     — handled at the model_runner level; runner not constructed.
"""

from __future__ import annotations

import copy
import inspect
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
import tqdm

from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    build_prefill_registry,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    compute_local_num_token_non_padded,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    BaseCudaGraphRunner,
    freeze_gc,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.utils import (
    resolve_prefill_backend,
)
from sglang.srt.model_executor.runner_backend_utils import (
    PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    BCG_FAILURE_HINT,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    TCPCG_FAILURE_HINT,
    set_tc_piecewise_forward_context,
)
from sglang.srt.model_executor.runner_utils.buffers import (
    PrefillInputBuffers,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.eagle_utils import get_draft_input_from_target_hidden_dim
from sglang.srt.utils import (
    get_available_gpu_memory,
    is_npu,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
)
from sglang.srt.utils.aiter import maybe_pre_warm_aiter_chip_info

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# A replay executes every padded token in its capture bucket. Sparse bucket
# lists can otherwise turn the lower launch overhead into substantially more
# model work than an exact-shape eager forward.
_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2


def prefill_failure_msg(backend_name: str) -> str:
    """Render PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG with a backend-specific
    numbered suggestion list. The runner is only constructed for BREAKABLE
    or TC_PIECEWISE; other values fall back to a generic OOM-style list."""
    if backend_name == Backend.BREAKABLE:
        hint = BCG_FAILURE_HINT
    elif backend_name == Backend.TC_PIECEWISE:
        hint = TCPCG_FAILURE_HINT
    else:
        hint = (
            "1. disable the prefill CUDA graph by --cuda-graph-backend-prefill=disabled\n"
            "2. if it is an OOM problem, set --mem-fraction-static to a smaller value "
            "(e.g., 0.8 or 0.7) or set --cuda-graph-max-bs-prefill to a smaller value "
            "(e.g., 2048)\n"
        )
    return PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG.format(
        backend=backend_name, suggestions=hint
    )


# Static prefill input tensors owned by captured-body backends (Breakable and
# Full). Each is a 1-D int64 tensor of length max_bs; captured graphs read
# these stable addresses during replay.
_PREFILL_STATIC_FIELDS = (
    "seq_lens",
    "extend_seq_lens",
    "extend_prefix_lens",
    "extend_start_loc",
    "req_pool_indices",
    "orig_seq_lens",
)


class PrefillCudaGraphRunner(BaseCudaGraphRunner):
    """Prefill-phase CUDA graph runner.

    Owns: PrefillInputBuffers, capture-num-tokens list, attention layers
    snapshot, and the pluggable self.backend. The backend handles capture
    + replay mechanics; this runner handles dummy ForwardBatch construction,
    buffer population, attention metadata init, and output slicing.
    """

    # DSA forces use_mha=False inside BCG capture/replay (see
    # DeepseekSparseAttnBackend.set_dsa_prefill_impl), so the captured graph
    # runs the sparse path for any prefix and the MHA-prefix replay ban does
    # not apply. Class-level default keeps partially-constructed instances
    # (unit tests via __new__) on the conservative ban.
    dsa_sparse_prefill_forced: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # --- model flags ----------------------------------------------
        self.quant_config = getattr(model_runner.model, "quant_config", None)
        self.is_multimodal = model_runner.model_config.is_multimodal
        # Classification/reward forwards branch on return_pooled_hidden_states;
        # capture must use the same flag value as replay for those models.
        self.capture_return_pooled_hidden_states = not model_runner.is_generation

        # --- prefill graph config -------------------------------------
        prefill_config = model_runner.server_args.cuda_graph_config.prefill
        self.prefill_backend_name = prefill_config.backend
        # bs in prefill carries the captured shape (token count for
        # tc_piecewise) — one shape knob per phase.
        capture_tokens = prefill_config.bs
        assert capture_tokens is not None, "cuda_graph_config[prefill].bs is not set"
        self.capture_num_tokens = sorted(capture_tokens)
        assert self.capture_num_tokens, "cuda_graph_config[prefill].bs is empty"

        # --- runner bounds --------------------------------------------
        self.max_num_tokens = max(self.capture_num_tokens)
        self.max_bs = model_runner.req_to_token_pool.size

        # --- capture modes --------------------------------------------
        self.capture_forward_mode = ForwardMode.EXTEND
        # Hidden-state capture mode cases:
        # - Breakable EAGLE draft: LAST.
        # - Breakable EAGLE target: FULL.
        # - Return-hidden-states or DFLASH: FULL.
        # - Otherwise: NULL.
        is_breakable_eagle = (
            self.prefill_backend_name == Backend.BREAKABLE
            and model_runner.spec_algorithm.is_eagle()
        )
        needs_full_hidden_states = (
            model_runner.server_args.enable_return_hidden_states
            or model_runner.spec_algorithm.is_dflash_family()
        )
        if is_breakable_eagle and model_runner.is_draft_worker:
            self.capture_hidden_mode = CaptureHiddenMode.LAST
        elif is_breakable_eagle or needs_full_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL
        else:
            self.capture_hidden_mode = CaptureHiddenMode.NULL

        self.mamba_track_enabled = self._is_mamba_track_enabled()

        # --- buffers ---------------------------------------------------
        self.buffers: PrefillInputBuffers = PrefillInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_tokens=self.max_num_tokens,
            cache_loc_dtype=self._cache_loc_dtype(),
            is_multimodal=self.is_multimodal,
            hidden_size=self.model_runner.model_config.hidden_size,
            dtype=self.model_runner.dtype,
            enable_mamba_track=self.mamba_track_enabled,
        )
        self.buffers.share_buffers()
        # Token-axis FB-shared slot registry adopting PrefillInputBuffers
        # storage; same physical tensors, stable data_ptr for capture vs
        # replay. Replaces populate_from_forward_batch on capture/replay paths.
        self.buffer_registry: CudaGraphBufferRegistry = build_prefill_registry(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_tokens,
            cache_loc_dtype=self._cache_loc_dtype(),
            is_multimodal=self.is_multimodal,
            hidden_size=self.model_runner.model_config.hidden_size,
            embed_dtype=self.model_runner.dtype,
            enable_mamba_track=self.mamba_track_enabled,
            enable_num_token_non_padded=enable_num_token_non_padded(),
            require_gathered_buffer=require_gathered_buffer(model_runner.server_args),
            enable_prefill_cp=(
                is_dsa_enable_prefill_cp() or is_mla_prefill_cp_enabled()
            ),
            source=self.buffers,
        )

        from sglang.srt.configs.model_config import is_deepseek_dsa

        self.dsa_sparse_prefill_forced = is_deepseek_dsa(
            self.model_runner.model_config.hf_config
        )

        self.attention_layers = self.model_runner.attention_layers
        self.mha_companion_layers = self.model_runner.mha_companion_layers
        self.has_mha_companion_layers = any(
            layer is not None for layer in self.mha_companion_layers
        )
        self.moe_layers = self.model_runner.moe_layers
        self.moe_fusions = self.model_runner.moe_fusions
        self.dsa_indexers = getattr(self.model_runner, "dsa_indexers", None)

        self.dp_size = model_runner.server_args.dp_size
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)

        # --- backend ---------------------------------------------------
        # TcPiecewise resolves by running a compile pass that calls back into
        # capture_prepare / _run_forward, so these fields must exist first.
        self._prefill_static_buffers: Optional[Dict[str, torch.Tensor]] = None
        self.static_draft_hidden_states: Optional[torch.Tensor] = None
        self.layer_model = None
        self._capture_req_slots = 1
        # Same rationale: _run_compile_pass runs a dummy _run_forward before
        # resolve_prefill_backend returns, and that forward reads
        # self._is_full_backend. The compile-pass backend is never Full, so
        # default False; the assignment below sets the real value once the
        # backend type is known.
        self._is_full_backend = False
        # TcPiecewise does its compile pass during backend construction.
        # Wrap only that path with the prefill CUDA graph failure hint.
        try:
            self.backend = resolve_prefill_backend(self)
        except RuntimeError as e:
            if self.prefill_backend_name != Backend.TC_PIECEWISE:
                raise
            raise RuntimeError(
                f"Capture prefill CUDA graph failed: {e}\n"
                f"{prefill_failure_msg(self.prefill_backend_name)}"
            ) from e

        self._is_full_backend = isinstance(self.backend, FullCudaGraphBackend)
        if self._is_full_backend:
            max_req = prefill_config.full_prefill_max_req
            if max_req is None:
                # Auto: scale request slots with the chunked prefill size.
                max_req = max(model_runner.server_args.chunked_prefill_size // 512, 1)
            self._capture_req_slots = min(max_req, self.max_bs)
            self._full_cg_seq_lens_cpu = torch.zeros(
                (self._capture_req_slots,), dtype=torch.int64, device="cpu"
            )
            with torch.device(self.device):
                self._prefill_static_buffers = {
                    name: torch.zeros((self.max_bs,), dtype=torch.int64)
                    for name in _PREFILL_STATIC_FIELDS
                }
        elif isinstance(self.backend, BreakableCudaGraphBackend):
            self._full_cg_seq_lens_cpu = None
            with torch.device(self.device):
                self._prefill_static_buffers = {
                    name: torch.zeros((self.max_bs,), dtype=torch.int64)
                    for name in _PREFILL_STATIC_FIELDS
                }
        else:
            self._full_cg_seq_lens_cpu = None

        # Static hidden_states buffer giving the captured graph a stable
        # address; load_batch refreshes it from live spec_info at replay.
        # Draft consumes aux-concatenated hidden states from the target
        # (e.g. EAGLE3 stacks 3 target layers), so capture the pre-reduction
        # width when the draft model exposes it.
        if (
            isinstance(self.backend, BreakableCudaGraphBackend)
            and model_runner.is_draft_worker
            and model_runner.spec_algorithm.is_eagle()
        ):
            hidden_dim = get_draft_input_from_target_hidden_dim(model_runner)
            with torch.device(self.device):
                self.static_draft_hidden_states = torch.zeros(
                    (self.max_num_tokens, hidden_dim),
                    dtype=model_runner.dtype,
                )

        # Some attention backends (e.g. DSV4) opt into a captured-metadata
        # contract under BCG: capture-time builds a per-bucket metadata
        # object the backend then refreshes in place at replay. We honor
        # the contract only when the backend is Breakable; FullCG and
        # TC_PIECEWISE use the eager init_forward_metadata path.
        if isinstance(self.backend, BreakableCudaGraphBackend):
            self.use_captured_attn_metadata = (
                model_runner.attn_backend.use_captured_forward_metadata_for_breakable_cuda_graph
            )
        else:
            self.use_captured_attn_metadata = False
        self.attn_metadata_buffers: Optional[Dict[int, object]] = (
            {} if self.use_captured_attn_metadata else None
        )

        # BCG and Full CG capture only the transformer body (layer_model.forward),
        # not the LM head + logits_processor — the eager tail keeps the captured
        # graph bs-invariant so req_slots is not bound by an (req_slots, vocab) buffer.
        if isinstance(self.backend, (BreakableCudaGraphBackend, FullCudaGraphBackend)):
            language_model = getattr(
                self.model_runner.model, "language_model", self.model_runner.model
            )
            if hasattr(language_model, "model") and hasattr(
                language_model.model, "layers"
            ):
                self.layer_model = language_model.model
            elif hasattr(language_model, "layers"):
                self.layer_model = language_model
            else:
                raise RuntimeError(
                    f"{type(self.backend).__name__} could not resolve inner "
                    f"layer_model on {type(language_model).__name__}; "
                    f"this backend is unsupported for this model architecture."
                )
            params = list(inspect.signature(self.layer_model.forward).parameters)
            self._input_embeds_arg_idx = (
                params.index("input_embeds") if "input_embeds" in params else None
            )

        # --- aiter chip info pre-warming (AMD) -------------------------
        maybe_pre_warm_aiter_chip_info()

        # --- capture --------------------------------------------------
        self.device_module.synchronize()
        self.model_runner.tp_group.barrier()
        self.capture()

        self.raw_num_tokens = 0
        self.raw_bs = 0

    def _is_mamba_track_enabled(self) -> bool:
        return (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and not self.model_runner.server_args.disable_radix_cache
            and self.model_runner.spec_algorithm.is_none()
        )

    def _cache_loc_dtype(self):
        return torch.int64 if not is_npu() else torch.int32

    def _next_token_logits_buffer(self, rows: int) -> Optional[torch.Tensor]:
        if not self.model_runner.pp_group.is_last_rank:
            return None
        graph_shared_output = self.model_runner.graph_shared_output
        # Fall back to eager logits when the shared buffer can't hold prefill rows.
        if graph_shared_output is None or rows > graph_shared_output.max_rows:
            return None
        return graph_shared_output.get_logits_buffer(
            self.model_runner.model_config.vocab_size, rows=rows
        )

    def _uses_eager_prefill_tail(self) -> bool:
        return self.prefill_backend_name in (Backend.BREAKABLE, Backend.FULL)

    def _prefill_logits_buffer_rows(self, forward_batch: ForwardBatch) -> int:
        if not forward_batch.return_logprob:
            return forward_batch.batch_size
        assert (
            self._uses_eager_prefill_tail()
        ), "Prefill return_logprob requires an eager logits tail."

        global_num_tokens = forward_batch.global_num_tokens_for_logprob_cpu
        if global_num_tokens is not None:
            dp_rank = get_parallel().attn_dp_rank
            return int(global_num_tokens[dp_rank if len(global_num_tokens) > 1 else 0])

        return sum(
            max(int(seq_len) - int(start_len), 1)
            for start_len, seq_len in zip(
                forward_batch.extend_logprob_start_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
            )
        )

    def _capture_num_token_non_padded(self, num_tokens: int) -> Optional[torch.Tensor]:
        if not self.buffer_registry.has_slot("num_token_non_padded"):
            return None

        buf = self.buffer_registry.get_slot("num_token_non_padded").buffer
        buf.fill_(num_tokens)
        if require_gathered_buffer(self.model_runner.server_args):
            local = compute_local_num_token_non_padded(
                global_num_token_non_padded=buf,
                num_tokens_per_dp=num_tokens,
            )
            buf.copy_(local)
        return buf

    def _get_layer_model_positions(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Mirror outer multimodal wrappers when BCG captures layer_model directly."""
        if forward_batch.mrope_positions is None:
            return forward_batch.positions

        model = self.model_runner.model
        if getattr(model, "is_mrope_enabled", False):
            return forward_batch.mrope_positions

        language_model = getattr(model, "language_model", None)
        if getattr(language_model, "is_mrope_enabled", False):
            return forward_batch.mrope_positions

        return forward_batch.positions

    @contextmanager
    def _prefill_forward_context(
        self,
        forward_batch: ForwardBatch,
        *,
        num_tokens: Optional[int] = None,
        raw_num_tokens: Optional[int] = None,
    ):
        with (
            forward_context(
                ForwardContext(attn_backend=self.model_runner.attn_backend)
            ),
            set_tc_piecewise_forward_context(
                forward_batch,
                self.attention_layers,
                self.quant_config,
                self.moe_layers,
                self.moe_fusions,
                dsa_indexers=self.dsa_indexers,
                mha_companion_layers=self.mha_companion_layers,
                num_tokens=num_tokens,
                raw_num_tokens=raw_num_tokens,
                full_graph=self._is_full_backend,
            ),
        ):
            yield

    @torch.no_grad()
    def _run_forward(self, forward_batch: ForwardBatch, num_tokens: int):
        """Run forward inside the prefill set_tc_piecewise_forward_context.

        BCG path: captures only the inner layer_model.forward (transformer
        stack), excluding the outer model.forward tail (logits_processor /
        pooler). The captured output is bs=1 hidden states; replay then runs
        the outer tail eagerly with live multi-req metadata.

        TC_PIECEWISE path: captures the outer model.forward; torch.compile
        FX-traces produce bs-invariant kernels.

        ``@torch.no_grad`` mirrors the decorator on the outer
        ``*ForCausalLM.forward``. For BCG, calling ``layer_model.forward``
        directly skips that decorator, so we apply it here — without it
        some MoE ``@torch.compile`` kernels (``torch.sum(out=...)``) fail
        dynamo with "out= doesn't support autograd", and mamba state ops
        can spuriously track gradients.
        """
        forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
        set_dp_buffer_len(
            forward_batch.global_dp_buffer_len,
            num_tokens,
            forward_batch.dp_padding_mode.is_max_len(),
            forward_batch.global_num_tokens_cpu,
        )
        set_is_extend_in_batch(False)

        with self._prefill_forward_context(forward_batch):
            if self._uses_eager_prefill_tail():
                # BCG / Full: capture the transformer body only.
                positions = self._get_layer_model_positions(forward_batch)
                return self.layer_model.forward(
                    forward_batch.input_ids,
                    positions,
                    forward_batch,
                    forward_batch.input_embeds,
                )
            # tc_piecewise: compile/capture the outer model.forward path.
            return self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

    def _run_dummy_forward(self, num_tokens: int) -> None:
        """Build a dummy ForwardBatch at this shape, init attn metadata,
        run forward once. Used by TcPiecewiseCudaGraphBackend.prepare
        for both the JIT-activate forward (single shape, before
        torch.compile install) and the compile-loop pass (every shape,
        inside enable_torch_compile_warmup).
        """
        fb, attn_backend = self.capture_prepare(num_tokens)
        attn_backend.init_forward_metadata(fb)
        self._run_forward(fb, num_tokens)

    def run_dummy_multimodal_deepstack_forward(
        self, language_model: torch.nn.Module, num_tokens: int
    ) -> bool:
        """Warm the tensor-valued deepstack branch before serving requests.

        The regular PCG dummy is text-only. Qwen3-VL only provides
        ``input_deepstack_embeds`` after visual encoding, so leaving this
        branch cold makes the first image request synchronously recompile the
        language model. The model/signature checks keep this a no-op for
        non-deepstack architectures.
        """
        if (
            "input_deepstack_embeds"
            not in inspect.signature(language_model.forward).parameters
        ):
            return False

        num_deepstack = getattr(self.model_runner.model, "num_deepstack_embeddings", 0)
        if num_deepstack <= 0:
            return False

        hidden_size = (
            getattr(getattr(language_model, "config", None), "hidden_size", None)
            or self.model_runner.model_config.hidden_size
        )
        fb, attn_backend = self.capture_prepare(num_tokens)
        attn_backend.init_forward_metadata(fb)
        deepstack_embeds = torch.zeros(
            (num_tokens, hidden_size * num_deepstack),
            dtype=self.model_runner.dtype,
            device=self.device,
        )
        torch._dynamo.maybe_mark_dynamic(deepstack_embeds, 0)

        fb.dp_local_start_pos = fb.dp_local_num_tokens = None
        set_dp_buffer_len(
            fb.global_dp_buffer_len,
            num_tokens,
            fb.dp_padding_mode.is_max_len(),
            fb.global_num_tokens_cpu,
        )
        set_is_extend_in_batch(False)

        with (
            forward_context(
                ForwardContext(attn_backend=self.model_runner.attn_backend)
            ),
            set_tc_piecewise_forward_context(
                fb,
                self.attention_layers,
                self.quant_config,
                self.moe_layers,
                self.moe_fusions,
                dsa_indexers=self.dsa_indexers,
            ),
        ):
            language_model.forward(
                fb.input_ids,
                self._get_layer_model_positions(fb),
                fb,
                input_embeds=fb.input_embeds,
                input_deepstack_embeds=deepstack_embeds,
            )
        return True

    def _has_inactive_dp_rank(self, forward_batch: ForwardBatch) -> bool:
        # DSV4 DP attention / DeepEP collectives need every DP rank to enter
        # the same replay path. Sparse-DP batches (one or more ranks with
        # zero local tokens) fall back to eager to avoid hanging ranks.
        global_num_tokens = forward_batch.global_num_tokens_cpu
        if global_num_tokens is None:
            return False
        return len(global_num_tokens) > 1 and any(
            int(num_tokens) == 0 for num_tokens in global_num_tokens
        )

    def _init_forward_metadata_for_capture(
        self, forward_batch: ForwardBatch, num_tokens: int
    ) -> None:
        """Capture-time metadata init for the BCG-with-captured-metadata
        contract. For opt-in backends (DSV4), call the BCG-specific entry
        and stash the returned per-bucket metadata object; otherwise fall
        back to the generic eager init that BCG/TC_PIECEWISE use today."""
        attn_backend = self.model_runner.attn_backend
        if not self.use_captured_attn_metadata:
            attn_backend.init_forward_metadata(forward_batch)
            return
        metadata = attn_backend.init_forward_metadata_for_breakable_cuda_graph_capture(
            forward_batch
        )
        assert self.attn_metadata_buffers is not None
        self.attn_metadata_buffers[num_tokens] = metadata

    def _prepare_forward_metadata_for_replay(
        self,
        forward_batch: ForwardBatch,
        static_forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> None:
        """Replay-time metadata refresh for the BCG-with-captured-metadata
        contract. For opt-in backends, refresh the stashed per-bucket
        metadata in place against the current batch; otherwise fall back
        to the generic eager init. Full CG instead refreshes the
        capture-stable wrapper state planned at capture time with the
        real seq_lens / prefix_lens; the captured kernels read the
        updated state at replay."""
        attn_backend = self.model_runner.attn_backend
        if self._is_full_backend:
            # Slot-padded shallow view: plan() must see exactly req_slots
            # entries (real values in [:bs], sentinels in [bs:req_slots]
            # already populated by replay_prepare).
            r = self._capture_req_slots
            bs = forward_batch.batch_size
            s = self._prefill_static_buffers
            self._full_cg_seq_lens_cpu.zero_()
            self._full_cg_seq_lens_cpu[:bs].copy_(forward_batch.seq_lens_cpu)
            padded_view = copy.copy(forward_batch)
            padded_view.batch_size = r
            padded_view.seq_lens = s["seq_lens"][:r]
            padded_view.seq_lens_cpu = self._full_cg_seq_lens_cpu
            padded_view.req_pool_indices = s["req_pool_indices"][:r]
            padded_view.extend_seq_lens = s["extend_seq_lens"][:r]
            padded_view.extend_prefix_lens = s["extend_prefix_lens"][:r]
            attn_backend.init_forward_metadata_out_graph(padded_view)
            return
        if not self.use_captured_attn_metadata:
            attn_backend.init_forward_metadata(forward_batch)
            return
        assert self.attn_metadata_buffers is not None
        metadata = self.attn_metadata_buffers[num_tokens]
        attn_backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            metadata,
            forward_batch,
            static_forward_batch=static_forward_batch,
        )

    def _has_unsupported_mha_prefix(self, forward_batch: ForwardBatch) -> bool:
        return (
            self.prefill_backend_name == Backend.BREAKABLE
            and self.has_mha_companion_layers
            and not self.dsa_sparse_prefill_forced
            and forward_batch.extend_prefix_lens_cpu is not None
            and any(forward_batch.extend_prefix_lens_cpu)
        )

    @staticmethod
    def _restore_mha_capture_state(forward_batch: ForwardBatch) -> None:
        """Restore Python state omitted from breakable graph segments."""
        forward_batch.mha_one_shot = True
        forward_batch.mha_return_lse = False
        forward_batch.set_attn_attend_prefix_cache(False)

    def schedule_batch_replay_eligible(self, batch) -> bool:
        """Rank-local replay eligibility, evaluated on the ScheduleBatch
        BEFORE the dp mlp-sync so it joins the min()-reduced
        can_run_breakable_cuda_graph consensus. Under DP attention every
        rank must reach the same replay-vs-eager decision: a lone eager
        rank re-derives its own attention path (e.g. MHA on DSA models)
        and its collectives mismatch the replaying ranks' captured ones,
        deadlocking the batch. Mirrors the rank-local checks of
        can_run_graph -- keep the two in sync.
        """
        if batch.input_embeds is not None:
            return False
        if (
            self.prefill_backend_name == Backend.BREAKABLE
            and self.has_mha_companion_layers
            and not self.dsa_sparse_prefill_forced
            and batch.prefix_lens is not None
            and any(batch.prefix_lens)
        ):
            return False
        num_tokens = batch.extend_num_tokens
        if num_tokens is None:
            return True
        if num_tokens > self.max_num_tokens:
            return False
        padded_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
        if padded_num_tokens > num_tokens * _MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR:
            return False
        return True

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        if self._is_full_backend and forward_batch.batch_size > self._capture_req_slots:
            return False
        if forward_batch.input_embeds is not None:
            return False
        if forward_batch.replace_embeds is not None:
            return False
        if self._has_unsupported_mha_prefix(forward_batch):
            return False
        # tc_piecewise captures with ForwardMode.EXTEND and spec_info=None.
        if forward_batch.forward_mode.is_target_verify():
            return False
        if forward_batch.capture_hidden_mode != self.capture_hidden_mode:
            return False
        # BCG-with-captured-metadata under DP attention: every rank must
        # have local tokens, and the batch must declare itself replayable.
        # These gates are no-ops for non-DP / non-opt-in paths because
        # global_num_tokens_cpu stays None.
        if self._has_inactive_dp_rank(forward_batch):
            return False
        if (
            forward_batch.global_num_tokens_cpu is not None
            and not forward_batch.can_run_dp_breakable_cuda_graph
        ):
            return False
        num_tokens = len(forward_batch.input_ids)
        if forward_batch.return_logprob and not self._uses_eager_prefill_tail():
            return False
        if num_tokens > self.max_num_tokens:
            return False
        padded_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
        if padded_num_tokens > num_tokens * _MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR:
            return False
        # No exact-shape check here: load_batch bucket-pads to the nearest
        # captured shape. The factor above only rejects replays whose padded
        # model work is disproportionate to the useful token count.
        #
        # Multi-req replay is supported by body-capture backends via the
        # layer_model.forward monkey-patch in replay(): the captured graph runs
        # the transformer stack, then the outer model.forward runs
        # logits_processor eagerly on top with live request metadata.
        return True

    def _build_capture_spec_info(self, num_tokens: int):
        if self.static_draft_hidden_states is None:
            return None
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        return EagleDraftInput(
            hidden_states=self.static_draft_hidden_states[:num_tokens],
        )

    def capture_prepare(self, num_tokens: int) -> tuple[ForwardBatch, AttentionBackend]:
        """Build a dummy prefill ForwardBatch for capture/warmup at this shape.

        Default tensor inputs are fresh literals; under a Breakable
        backend, we swap in slices of our static buffers so captured
        segments read from stable addresses.

        Returns ``(forward_batch, attn_backend)`` to mirror decode's
        capture_prepare signature.
        """
        bs = self._capture_req_slots
        # Slot 0 carries num_tokens; slots 1..bs-1 are zero-length sentinels.
        lens_cpu = [num_tokens] + [0] * (bs - 1)
        start_loc_cpu = [0] + [num_tokens] * (bs - 1)

        with torch.device(self.device):
            shape_inputs = {
                "req_pool_indices": torch.arange(bs, device=self.device),
                "seq_lens": torch.tensor(lens_cpu, device=self.device),
                "orig_seq_lens": torch.tensor(lens_cpu, device=self.device),
                "extend_seq_lens": torch.tensor(lens_cpu, device=self.device),
                "extend_prefix_lens": torch.zeros((bs,), dtype=torch.int64),
                "extend_start_loc": torch.tensor(start_loc_cpu, device=self.device),
            }
        if self._prefill_static_buffers is not None:
            s = self._prefill_static_buffers
            s["seq_lens"][:bs].copy_(shape_inputs["seq_lens"])
            s["extend_seq_lens"][:bs].copy_(shape_inputs["extend_seq_lens"])
            s["extend_prefix_lens"][:bs].zero_()
            s["extend_start_loc"][:bs].copy_(shape_inputs["extend_start_loc"])
            s["req_pool_indices"][:bs].copy_(
                torch.arange(bs, device=s["req_pool_indices"].device)
            )
            s["orig_seq_lens"][:bs].copy_(shape_inputs["orig_seq_lens"])
            for name in _PREFILL_STATIC_FIELDS:
                shape_inputs[name] = s[name][:bs]

        registry = self.buffer_registry

        def _slot(name):
            return registry.get_slot(name).slice_for(bs, num_tokens)

        if self.require_mlp_tp_gather:
            global_num_tokens_cpu = [num_tokens] * self.dp_size
        elif self.require_attn_tp_gather:
            global_num_tokens_cpu = [num_tokens]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu, dtype=torch.int32, device=self.device
            )
            global_num_tokens_gpu = num_tokens_tensor
            global_num_tokens_for_logprob_gpu = num_tokens_tensor
        else:
            global_dp_buffer_len = None
            global_num_tokens_gpu = None
            global_num_tokens_for_logprob_gpu = None

        with torch.device(self.device):
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=bs,
                input_ids=_slot("input_ids"),
                input_embeds=(
                    _slot("input_embeds") if registry.has_slot("input_embeds") else None
                ),
                req_pool_indices=shape_inputs["req_pool_indices"],
                seq_lens=shape_inputs["seq_lens"],
                next_token_logits_buffer=self._next_token_logits_buffer(bs),
                orig_seq_lens=shape_inputs["orig_seq_lens"],
                seq_lens_cpu=torch.tensor(lens_cpu, device="cpu"),
                out_cache_loc=_slot("out_cache_loc"),
                seq_lens_sum=num_tokens,
                mamba_track_indices=(
                    _slot("mamba_track_indices")
                    if registry.has_slot("mamba_track_indices")
                    else None
                ),
                mamba_track_mask=(
                    _slot("mamba_track_mask")
                    if registry.has_slot("mamba_track_mask")
                    else None
                ),
                mamba_track_seqlens=(
                    _slot("mamba_track_seqlens")
                    if registry.has_slot("mamba_track_seqlens")
                    else None
                ),
                encoder_lens=None,
                return_logprob=False,
                extend_num_tokens=num_tokens,
                extend_seq_lens=shape_inputs["extend_seq_lens"],
                extend_prefix_lens=shape_inputs["extend_prefix_lens"],
                extend_start_loc=shape_inputs["extend_start_loc"],
                extend_prefix_lens_cpu=[0] * bs,
                extend_seq_lens_cpu=list(lens_cpu),
                extend_logprob_start_lens_cpu=list(lens_cpu),
                positions=_slot("positions"),
                global_num_tokens_gpu=global_num_tokens_gpu,
                global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
                global_num_tokens_cpu=global_num_tokens_cpu,
                dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
                global_dp_buffer_len=global_dp_buffer_len,
                mrope_positions=(
                    _slot("mrope_positions")
                    if registry.has_slot("mrope_positions")
                    else None
                ),
                spec_algorithm=None,
                spec_info=self._build_capture_spec_info(num_tokens),
                # Use self.capture_hidden_mode so dflash spec (which needs
                # FULL aux hidden states) captures with the right mode.
                # Ported from main #27468.
                capture_hidden_mode=self.capture_hidden_mode,
                num_token_non_padded=self._capture_num_token_non_padded(num_tokens),
                num_token_non_padded_cpu=num_tokens,
                global_forward_mode=ForwardMode.EXTEND,
                lora_ids=None,
                return_pooled_hidden_states=self.capture_return_pooled_hidden_states,
            )
            self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)
        return forward_batch, self.model_runner.attn_backend

    def capture(self) -> None:
        # Warm up + autotune kernels once before capture (run-once across the
        # decode + prefill runners; see BaseRunner.warmup).
        self.warmup()
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
            with graph_capture() as graph_capture_context:
                self.stream = graph_capture_context.stream
                with self.backend.capture_session(self.stream):
                    self._capture_one_stream()

    def _capture_one_stream(self) -> None:
        avail_mem = get_available_gpu_memory(
            self.model_runner.device,
            self.model_runner.gpu_id,
            empty_cache=False,
        )
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_num_tokens)))
            if get_parallel().tp_rank == 0
            else reversed(self.capture_num_tokens)
        )
        for num_tokens in capture_range:
            if get_parallel().tp_rank == 0:
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
        delegate to backend. size is the prefill token count.
        """
        num_tokens = size
        forward_batch, attn_backend = self.capture_prepare(num_tokens)
        if self._is_full_backend:
            attn_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
        else:
            self._init_forward_metadata_for_capture(forward_batch, num_tokens)

        def run_once():
            return self._run_forward(forward_batch, num_tokens)

        # Main's monolithic BCG runner never invokes
        # on_after_cuda_graph_warmup between warmup iterations — the BCG
        # contract is to keep warmup state untouched and let
        # init_forward_metadata_in_graph (recorded inside the captured
        # forward) do any raw->full upgrade. cg-refactor's runner_backend
        # abstraction exposes a post_warmup_hook for backends that need
        # workspace cleanup between iterations; suppress it for BCG so
        # DSV4's hook (which restores forward_metadata to a stale
        # _current_capture_raw left over from decode CG capture) doesn't
        # corrupt warmup iter 2's metadata read.
        if isinstance(self.backend, BreakableCudaGraphBackend):
            post_warmup_hook = None
        else:
            post_warmup_hook = getattr(attn_backend, "on_after_cuda_graph_warmup", None)
        self.backend.capture_one(
            ShapeKey(size=num_tokens),
            run_once,
            # DP padding can install capture-only tensors on this dummy batch;
            # BCG retains it so their recorded addresses remain valid.
            capture_inputs=forward_batch,
            post_warmup_hook=post_warmup_hook,
        )

    def load_batch(self, forward_batch: ForwardBatch, **kwargs) -> ForwardBatch:
        """Pad, populate static buffers, and build the static_forward_batch
        the model code reads during replay.
        """
        num_tokens = len(forward_batch.input_ids)
        static_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
        self.raw_num_tokens = num_tokens

        bs = forward_batch.batch_size
        self.raw_bs = bs

        self.buffer_registry.fill_from(
            forward_batch,
            raw_bs=bs,
            padded_bs=bs,
            raw_num_tokens=num_tokens,
            padded_num_tokens=static_num_tokens,
        )

        registry = self.buffer_registry

        def _slot(name):
            return registry.get_slot(name).slice_for(bs, static_num_tokens)

        mamba_track_indices = (
            _slot("mamba_track_indices")
            if registry.has_slot("mamba_track_indices")
            else None
        )
        mamba_track_mask = (
            _slot("mamba_track_mask") if registry.has_slot("mamba_track_mask") else None
        )
        mamba_track_seqlens = (
            _slot("mamba_track_seqlens")
            if registry.has_slot("mamba_track_seqlens")
            else None
        )

        input_ids = _slot("input_ids")
        input_embeds = (
            _slot("input_embeds") if registry.has_slot("input_embeds") else None
        )
        positions = _slot("positions")
        out_cache_loc = _slot("out_cache_loc")
        mrope_positions = (
            _slot("mrope_positions")
            if registry.has_slot("mrope_positions")
            and forward_batch.mrope_positions is not None
            else None
        )
        num_token_non_padded = (
            _slot("num_token_non_padded")
            if registry.has_slot("num_token_non_padded")
            else forward_batch.num_token_non_padded
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
            next_token_logits_buffer=self._next_token_logits_buffer(
                self._prefill_logits_buffer_rows(forward_batch)
            ),
            orig_seq_lens=forward_batch.orig_seq_lens,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=forward_batch.seq_lens_sum,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=mamba_track_seqlens,
            encoder_lens=forward_batch.encoder_lens,
            return_logprob=(
                forward_batch.return_logprob
                if isinstance(self.backend, BreakableCudaGraphBackend)
                else False
            ),
            is_prefill_only=forward_batch.is_prefill_only,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            extend_start_loc=forward_batch.extend_start_loc,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
            token_ids_logprobs=forward_batch.token_ids_logprobs,
            multi_item_delimiter_indices=forward_batch.multi_item_delimiter_indices,
            extend_num_tokens=forward_batch.extend_num_tokens,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            positions=positions,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
            global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,
            dp_padding_mode=forward_batch.dp_padding_mode,
            global_dp_buffer_len=forward_batch.global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=forward_batch.spec_algorithm,
            spec_info=forward_batch.spec_info,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            num_token_non_padded=num_token_non_padded,
            num_token_non_padded_cpu=forward_batch.num_token_non_padded_cpu,
            global_forward_mode=pcg_global_forward_mode,
            lora_ids=forward_batch.lora_ids,
            sampling_info=forward_batch.sampling_info,
            mm_inputs=forward_batch.mm_inputs,
            temperature=forward_batch.temperature,
            top_p=forward_batch.top_p,
            dimensions=forward_batch.dimensions,
            return_pooled_hidden_states=(
                self.capture_return_pooled_hidden_states
                or forward_batch.return_pooled_hidden_states
            ),
        )
        if self._is_full_backend:
            forward_batch.next_token_logits_buffer = (
                static_forward_batch.next_token_logits_buffer
            )

        if (
            isinstance(self.backend, BreakableCudaGraphBackend)
            and self.has_mha_companion_layers
        ):
            self._restore_mha_capture_state(static_forward_batch)

        # Under Breakable / Full, copy serving-time values into the static
        # buffers so the addresses captured segments hold stay live with
        # current data.
        if self._prefill_static_buffers is not None:
            bs = forward_batch.batch_size
            s = self._prefill_static_buffers
            s["seq_lens"][:bs].copy_(forward_batch.seq_lens)
            s["extend_seq_lens"][:bs].copy_(forward_batch.extend_seq_lens)
            s["extend_prefix_lens"][:bs].copy_(forward_batch.extend_prefix_lens)
            s["extend_start_loc"][:bs].copy_(forward_batch.extend_start_loc)
            s["req_pool_indices"][:bs].copy_(forward_batch.req_pool_indices)
            if forward_batch.orig_seq_lens is not None:
                s["orig_seq_lens"][:bs].copy_(forward_batch.orig_seq_lens)
            if self._is_full_backend and bs < self._capture_req_slots:
                # Sentinel tail for slots [bs:req_slots]: the captured graph
                # reads all req_slots entries (e.g. the logits-processor
                # cumsum), so stale values from the previous replay must be
                # cleared. Zero lengths make the sentinels no-ops;
                # extend_start_loc sentinels sit at the flat end of the real
                # tokens.
                r = self._capture_req_slots
                s["seq_lens"][bs:r].zero_()
                s["extend_seq_lens"][bs:r].zero_()
                s["extend_prefix_lens"][bs:r].zero_()
                s["extend_start_loc"][bs:r].fill_(self.raw_num_tokens)
                s["req_pool_indices"][bs:r].zero_()
                s["orig_seq_lens"][bs:r].zero_()

        # Refresh the static buffer the captured graph reads from.
        if (
            self.static_draft_hidden_states is not None
            and forward_batch.spec_info is not None
        ):
            self.static_draft_hidden_states[:num_tokens].copy_(
                forward_batch.spec_info.hidden_states
            )

        self._prepare_forward_metadata_for_replay(
            forward_batch, static_forward_batch, static_num_tokens
        )

        self._static_num_tokens = static_num_tokens
        return static_forward_batch

    def _execute_body_capture(
        self,
        forward_batch: ForwardBatch,
        static_forward_batch: ForwardBatch,
        static_num_tokens: int,
        raw_num_tokens: int,
        **kwargs,
    ):
        # BCG / Full: replay the captured body, run the LM head +
        # logits_processor eagerly.
        shape_key = ShapeKey(size=self._static_num_tokens)
        full_path = self._is_full_backend
        static_n = self._static_num_tokens
        ie_idx = self._input_embeds_arg_idx

        def replay_layer_forward(*args, **layer_kwargs):
            # The captured body graph reads activations from the static
            # input_embeds slot. The outer model.forward (run eagerly)
            # passes the live embeddings into layer_model.forward as the
            # 4th positional arg (or input_embeds kwarg): for multimodal
            # batches these are the composed text+vision embeds, for
            # text-only batches they are get_input_embeddings()(input_ids).
            # Copy them into the slot before replay so the graph sees the
            # current request's embeddings (mirrors main's BCG closure).
            if self.buffer_registry.has_slot("input_embeds"):
                ie = layer_kwargs.get("input_embeds")
                if ie is None and ie_idx is not None and len(args) > ie_idx:
                    ie = args[ie_idx]
                if ie is not None:
                    self.buffer_registry.get_slot("input_embeds").slice_for(
                        1, static_n
                    )[: ie.shape[0]].copy_(ie)
            hs = self.backend.replay(shape_key, static_forward_batch, **kwargs)
            return hs[:raw_num_tokens] if full_path else hs

        original_layer_forward = self.layer_model.forward
        self.layer_model.forward = replay_layer_forward
        # For Full, run the eager tail against the raw user-facing batch so it
        # uses real request metadata instead of padded slots. BCG has no
        # request-slot padding, so static_forward_batch is already the serving batch.
        tail_batch = forward_batch if full_path else static_forward_batch
        try:
            with self._prefill_forward_context(
                static_forward_batch,
                num_tokens=static_num_tokens,
                raw_num_tokens=raw_num_tokens,
            ):
                return self.model_runner.model.forward(
                    tail_batch.input_ids,
                    tail_batch.positions,
                    tail_batch,
                    **kwargs,
                )
        finally:
            self.layer_model.forward = original_layer_forward

    def _execute_tc_piecewise(
        self,
        static_forward_batch: ForwardBatch,
        static_num_tokens: int,
        raw_num_tokens: int,
        **kwargs,
    ):
        with self._prefill_forward_context(
            static_forward_batch,
            num_tokens=static_num_tokens,
            raw_num_tokens=raw_num_tokens,
        ):
            return self.backend.replay(
                ShapeKey(size=self._static_num_tokens),
                static_forward_batch,
                **kwargs,
            )

    def _trim_logits_output(
        self, output: LogitsProcessorOutput
    ) -> LogitsProcessorOutput:
        # Preserve mm_input_embeds for speculative decoding.
        mm_input_embeds = None
        if (
            self.model_runner.spec_algorithm.is_speculative()
            and output.mm_input_embeds is not None
        ):
            mm_input_embeds = output.mm_input_embeds[: self.raw_num_tokens]
        logits_rows = self.raw_bs if self._is_full_backend else self.raw_num_tokens
        return LogitsProcessorOutput(
            next_token_logits=(
                output.next_token_logits[:logits_rows]
                if output.next_token_logits is not None
                else None
            ),
            hidden_states=(
                output.hidden_states[: self.raw_num_tokens]
                if output.hidden_states is not None
                else None
            ),
            input_token_logprobs=output.input_token_logprobs,
            input_top_logprobs_val=output.input_top_logprobs_val,
            input_top_logprobs_idx=output.input_top_logprobs_idx,
            input_token_ids_logprobs_val=output.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=output.input_token_ids_logprobs_idx,
            mm_input_embeds=mm_input_embeds,
        )

    def _finalize_execute_output(
        self, output
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        if isinstance(output, LogitsProcessorOutput):
            return self._trim_logits_output(output)
        if isinstance(output, EmbeddingPoolerOutput):
            return output
        assert isinstance(output, PPProxyTensors)
        raise NotImplementedError(
            "PPProxyTensors is not supported in PrefillCudaGraphRunner yet."
        )

    def execute(
        self, forward_batch: ForwardBatch, **kwargs
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        with self.backend.replay_session():
            static_forward_batch = self.load_batch(forward_batch, **kwargs)
            static_num_tokens = len(static_forward_batch.input_ids)
            raw_num_tokens = self.raw_num_tokens

            if self._uses_eager_prefill_tail():
                output = self._execute_body_capture(
                    forward_batch,
                    static_forward_batch,
                    static_num_tokens,
                    raw_num_tokens,
                    **kwargs,
                )
            else:
                output = self._execute_tc_piecewise(
                    static_forward_batch,
                    static_num_tokens,
                    raw_num_tokens,
                    **kwargs,
                )
            return self._finalize_execute_output(output)
