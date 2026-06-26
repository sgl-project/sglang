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
  - "tc_piecewise"     — default, TcPiecewiseCudaGraphBackend: torch.compile
                      wraps the model; per-shape graphs live in
                      torch.compile's internal cache. Multi-batch supported.
  - "breakable" — BreakableCudaGraphBackend: segmented capture (no
                      torch.compile). Captures with bs=1; rejects multi-req
                      prefill in can_run_graph.
  - "full"      — rejected at config validation; not supported for prefill.
  - "disabled"  — handled at the model_runner level — runner not
                      constructed.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Union

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
from sglang.srt.model_executor.runner_backend.utils import (
    resolve_prefill_backend,
)
from sglang.srt.model_executor.runner_backend_utils import (
    PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.srt.model_executor.runner_utils.buffers import (
    PrefillInputBuffers,
)
from sglang.srt.speculative.eagle_utils import get_draft_input_from_target_hidden_dim
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_bool_env_var,
    is_hip,
    is_npu,
    require_attn_tp_gather,
    require_mlp_tp_gather,
)

# Suppress Dynamo warning about tracing through lru_cache-wrapped functions.
warnings.filterwarnings("ignore", message=".*lru_cache.*", module="torch._dynamo")
logger = logging.getLogger(__name__)

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

# Names of the static prefill input tensors a Breakable-backed prefill
# runner owns. Each is a 1-D int64 tensor of length max_bs; captured
# Breakable segments read from these stable addresses.
_PREFILL_STATIC_FIELDS = (
    "seq_lens",
    "extend_seq_lens",
    "extend_prefix_lens",
    "extend_start_loc",
    "req_pool_indices",
    "orig_seq_lens",
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner


class PrefillCudaGraphRunner(BaseCudaGraphRunner):
    """Prefill-phase CUDA graph runner.

    Owns: PrefillInputBuffers, capture-num-tokens list, attention layers
    snapshot, and the pluggable self.backend. The backend handles capture
    + replay mechanics; this runner handles dummy ForwardBatch construction,
    buffer population, attention metadata init, and output slicing.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # --- core state ------------------------------------------------
        self.quant_config = getattr(self.model_runner.model, "quant_config", None)
        self.is_multimodal = model_runner.is_multimodal
        # Classification/reward forwards branch on return_pooled_hidden_states;
        # capture must use the same flag value as replay for those models.
        self.capture_return_pooled_hidden_states = not model_runner.is_generation

        # --- bucket sizes ---------------------------------------------
        # bs in prefill carries the captured shape (token count for
        # tc_piecewise) — one shape knob per phase.
        capture_tokens = model_runner.server_args.cuda_graph_config.prefill.bs
        assert capture_tokens is not None, "cuda_graph_config[prefill].bs is not set"
        self.capture_num_tokens = sorted(capture_tokens)
        self.max_num_tokens = (
            max(self.capture_num_tokens) if self.capture_num_tokens else 8192
        )
        self.max_bs = model_runner.req_to_token_pool.size

        self.capture_forward_mode = ForwardMode.EXTEND
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        # If returning hidden states is enabled, or if speculative prefill
        # needs aux hidden states (DFLASH), capture the FULL variant up front.
        # Ported from main #27468.
        if (
            model_runner.server_args.enable_return_hidden_states
            or model_runner.spec_algorithm.is_dflash()
        ):
            self.capture_hidden_mode = CaptureHiddenMode.FULL
        # EAGLE captures FULL hidden states for the target and LAST for the
        # draft (can_run_graph rejects on mismatch). BCG only; tc_piecewise
        # EAGLE is routed to eager in ModelRunner.init_prefill_cuda_graph.
        _cg_cfg = model_runner.server_args.cuda_graph_config
        _prefill_backend_name = (
            _cg_cfg.prefill.backend if _cg_cfg is not None else Backend.TC_PIECEWISE
        )
        if (
            _prefill_backend_name == Backend.BREAKABLE
            and model_runner.spec_algorithm.is_eagle()
        ):
            self.capture_hidden_mode = (
                CaptureHiddenMode.LAST
                if model_runner.is_draft_worker
                else CaptureHiddenMode.FULL
            )

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
            # Register the multimodal input_embeds slot for every prefill
            # backend (default True). The slot is only added when is_multimodal,
            # so text-only models are unaffected. Both tc_piecewise (outer MM
            # wrapper passes composed input_embeds as an argument) and breakable
            # (captures the input_embeds path; general_mm_embed_routine fills the
            # slot) need it, otherwise the captured graph re-embeds input_ids and
            # drops the scattered vision embeddings.
            source=self.buffers,
        )

        self.attention_layers = self.model_runner.attention_layers
        self.moe_layers = self.model_runner.moe_layers
        self.moe_fusions = self.model_runner.moe_fusions
        self.dsa_indexers = getattr(self.model_runner, "dsa_indexers", None)

        self.dp_size = model_runner.server_args.dp_size
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)

        # --- backend ---------------------------------------------------
        # When the backend is Breakable, captured segments need stable
        # tensor addresses, so we own a set of static int64 buffers here
        # and rebind them into capture-time dummy inputs / replay-time
        # serving inputs below. Other backends don't need this.
        # Initialize the slot to None BEFORE constructing the backend:
        # TcPiecewise runs its compile pass during __init__ which calls
        # _run_dummy_forward -> capture_prepare, and capture_prepare reads
        # self._prefill_static_buffers and self.static_draft_hidden_states.
        # self.layer_model has the same ordering requirement: _run_forward
        # checks `self.layer_model is not None` to decide whether to call
        # the inner stack or outer model.forward, and that check fires
        # inside TcPiecewise's _run_compile_pass before backend resolution
        # returns.
        self._prefill_static_buffers: Optional[Dict[str, torch.Tensor]] = None
        self.static_draft_hidden_states: Optional[torch.Tensor] = None
        self.layer_model = None
        try:
            self.backend = resolve_prefill_backend(self)
        except RuntimeError as e:
            if _prefill_backend_name == Backend.TC_PIECEWISE:
                raise Exception(
                    f"Capture prefill CUDA graph failed: {e}\n"
                    f"{PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG}"
                )
            raise
        if isinstance(self.backend, BreakableCudaGraphBackend):
            with torch.device(self.device):
                self._prefill_static_buffers = {
                    name: torch.zeros((self.max_bs,), dtype=torch.int64)
                    for name in _PREFILL_STATIC_FIELDS
                }

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

        # --- BCG: resolve inner layer_model for capture/replay --------
        # BCG captures only the inner transformer stack (layer_model.forward)
        # — not the outer model.forward. The outer's tail (logits_processor /
        # pooler) has bs-shaped kernels that would bake bs=1 into the captured
        # graph and break multi-req replay. At replay, we monkey-patch
        # layer_model.forward to replay the captured graph and return the
        # captured hidden states; the outer model.forward then runs
        # logits_processor eagerly on top with the live multi-req metadata.
        # Mirrors main's BreakableCudaGraphRunner. (Slot pre-init lives
        # above next to _prefill_static_buffers — TcPiecewise's compile
        # pass runs during backend construction and reads self.layer_model.)
        if isinstance(self.backend, BreakableCudaGraphBackend):
            language_model = getattr(
                self.model_runner.model, "language_model", self.model_runner.model
            )
            if hasattr(language_model, "model") and hasattr(
                language_model.model, "layers"
            ):
                self.layer_model = language_model.model
            else:
                raise RuntimeError(
                    f"BCG could not resolve inner layer_model on "
                    f"{type(language_model).__name__}; BCG is unsupported for "
                    f"this model architecture."
                )

        # --- aiter chip info pre-warming (AMD) -------------------------
        if _use_aiter:
            self._pre_warm_aiter_chip_info()

        # --- capture --------------------------------------------------
        self.device_module.synchronize()
        self.model_runner.tp_group.barrier()
        try:
            self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture prefill CUDA graph failed: {e}\n"
                f"{PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

        self.raw_num_tokens = 0

    def _is_mamba_track_enabled(self) -> bool:
        return (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and not self.model_runner.server_args.disable_radix_cache
            and self.model_runner.spec_algorithm.is_none()
        )

    def _cache_loc_dtype(self):
        return torch.int64 if not is_npu() else torch.int32

    _aiter_chip_info_cached = False

    @classmethod
    def _pre_warm_aiter_chip_info(cls):
        """Pre-populate aiter chip info env vars before CUDA graph capture.

        aiter's get_cu_num_custom_op and get_gfx_custom_op call
        subprocess.run(rocminfo) to query GPU info. During CUDA graph capture
        the GPU context is locked, so rocminfo hangs indefinitely. Pre-calling
        them here caches the results as environment variables so the subprocess
        is never invoked during capture. Only runs once per process.
        """
        if cls._aiter_chip_info_cached:
            return
        cls._aiter_chip_info_cached = True

        import os

        try:
            from aiter.jit.utils.chip_info import get_cu_num, get_gfx

            if not os.environ.get("CU_NUM"):
                cu_num = get_cu_num()
                os.environ["CU_NUM"] = str(cu_num)
                logger.info(f"Pre-warmed aiter CU_NUM={cu_num}")

            if not os.environ.get("GPU_ARCHS"):
                gfx = get_gfx()
                os.environ["GPU_ARCHS"] = gfx
                logger.info(f"Pre-warmed aiter GPU_ARCHS={gfx}")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to pre-warm aiter chip info: {e}")

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
            ),
        ):
            if self.layer_model is not None:
                return self.layer_model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    forward_batch.input_embeds,
                )
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
        to the generic eager init."""
        attn_backend = self.model_runner.attn_backend
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

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        if forward_batch.input_embeds is not None:
            return False
        if forward_batch.replace_embeds is not None:
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
        if forward_batch.return_logprob:
            for start_len, seq_len in zip(
                forward_batch.extend_logprob_start_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
            ):
                if start_len is not None and start_len < seq_len:
                    return False
        if num_tokens > self.max_num_tokens:
            return False
        # No backend-level shape check here: load_batch bucket-pads
        # num_tokens up to the nearest captured shape, so eligibility is
        # bounded by num_tokens <= self.max_num_tokens (already
        # checked above), not by exact shape membership.
        #
        # Multi-req replay is supported by BCG via the layer_model.forward
        # monkey-patch in replay(): the captured bs=1 graph runs the
        # transformer stack, then the outer model.forward runs
        # logits_processor eagerly on top with live multi-req metadata.
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
        if self._prefill_static_buffers is not None:
            s = self._prefill_static_buffers
            s["seq_lens"][:bs].fill_(num_tokens)
            s["extend_seq_lens"][:bs].fill_(num_tokens)
            s["extend_prefix_lens"][:bs].zero_()
            s["extend_start_loc"][:bs].zero_()
            s["req_pool_indices"][:bs].copy_(
                torch.arange(bs, device=s["req_pool_indices"].device)
            )
            s["orig_seq_lens"][:bs].fill_(num_tokens)
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
                next_token_logits_buffer=None,
                orig_seq_lens=shape_inputs["orig_seq_lens"],
                seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
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
                extend_prefix_lens_cpu=torch.tensor([0], device="cpu"),
                extend_seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                extend_logprob_start_lens_cpu=torch.tensor([num_tokens], device="cpu"),
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
                num_token_non_padded=None,
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
        delegate to backend. size is the prefill token count.
        """
        num_tokens = size
        forward_batch, attn_backend = self.capture_prepare(num_tokens)
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
            dummies=None,
            post_warmup_hook=post_warmup_hook,
        )

    def load_batch(self, forward_batch: ForwardBatch, **kwargs) -> ForwardBatch:
        """Pad, populate static buffers, and build the static_forward_batch
        the model code reads during replay.
        """
        buffers = self.buffers
        num_tokens = len(forward_batch.input_ids)
        static_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)
        self.raw_num_tokens = num_tokens

        bs = forward_batch.batch_size

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
            out_cache_loc=out_cache_loc,
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
            temperature=forward_batch.temperature,
            top_p=forward_batch.top_p,
            dimensions=forward_batch.dimensions,
            return_pooled_hidden_states=(
                self.capture_return_pooled_hidden_states
                or forward_batch.return_pooled_hidden_states
            ),
        )

        # Under Breakable, copy serving-time values into the static
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

    def execute(
        self, forward_batch: ForwardBatch, **kwargs
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        with self.backend.replay_session():
            static_forward_batch = self.load_batch(forward_batch, **kwargs)
            static_num_tokens = len(static_forward_batch.input_ids)
            raw_num_tokens = self.raw_num_tokens

            if self.layer_model is not None:
                # BCG path. The captured graph is a bs=1 replay of
                # layer_model.forward. Monkey-patch layer_model.forward to
                # call backend.replay (which fires the captured graph and
                # returns the captured hidden_states), then drive the outer
                # model.forward eagerly with the live multi-req
                # static_forward_batch. The outer's logits_processor /
                # pooler then runs on top with live multi-req metadata.
                shape_key = ShapeKey(size=self._static_num_tokens)
                static_n = self._static_num_tokens

                def replay_layer_forward(*args, **layer_kwargs):
                    # The captured BCG graph reads activations from the static
                    # input_embeds slot. The outer model.forward (run eagerly)
                    # passes the live embeddings into layer_model.forward as the
                    # 4th positional arg (or input_embeds kwarg): for multimodal
                    # batches these are the composed text+vision embeds, for
                    # text-only batches they are get_input_embeddings()(input_ids).
                    # Copy them into the slot before replay so the graph sees the
                    # current request's embeddings (mirrors main's BCG closure).
                    if self.buffer_registry.has_slot("input_embeds"):
                        ie = layer_kwargs.get("input_embeds")
                        if ie is None and len(args) > 3:
                            ie = args[3]
                        if ie is not None:
                            self.buffer_registry.get_slot("input_embeds").slice_for(
                                1, static_n
                            ).copy_(ie[:static_n])
                    return self.backend.replay(
                        shape_key, static_forward_batch, **kwargs
                    )

                original_layer_forward = self.layer_model.forward
                self.layer_model.forward = replay_layer_forward
                try:
                    with (
                        forward_context(
                            ForwardContext(attn_backend=self.model_runner.attn_backend)
                        ),
                        set_tc_piecewise_forward_context(
                            static_forward_batch,
                            self.attention_layers,
                            self.quant_config,
                            self.moe_layers,
                            self.moe_fusions,
                            dsa_indexers=self.dsa_indexers,
                            num_tokens=static_num_tokens,
                            raw_num_tokens=raw_num_tokens,
                        ),
                    ):
                        output = self.model_runner.model.forward(
                            static_forward_batch.input_ids,
                            static_forward_batch.positions,
                            static_forward_batch,
                            **kwargs,
                        )
                finally:
                    self.layer_model.forward = original_layer_forward
            else:
                # TC_PIECEWISE path. backend.replay calls the compiled
                # outer model.forward directly (torch.compile handles
                # multi-req via bs-invariant FX-traced kernels).
                with (
                    forward_context(
                        ForwardContext(attn_backend=self.model_runner.attn_backend)
                    ),
                    set_tc_piecewise_forward_context(
                        static_forward_batch,
                        self.attention_layers,
                        self.quant_config,
                        self.moe_layers,
                        self.moe_fusions,
                        dsa_indexers=self.dsa_indexers,
                        num_tokens=static_num_tokens,
                        raw_num_tokens=raw_num_tokens,
                    ),
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
