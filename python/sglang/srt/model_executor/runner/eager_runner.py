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
"""No-cuda-graph phase runner; the eager dual of BaseCudaGraphRunner."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.cp.utils import (
    cp_gather_after_forward,
    cp_split_before_forward,
    is_cp_v2_active,
    prepare_cp_forward,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_eager_registry,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
    set_tc_piecewise_forward_context,
)
from sglang.srt.utils import is_hip, require_mlp_tp_gather
from sglang.srt.utils.common import ceil_align, require_mlp_sync

logger = logging.getLogger(__name__)

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.model_runner import ModelRunner


class EagerRunner(BaseRunner):
    def __init__(self, model_runner: ModelRunner) -> None:
        super().__init__(model_runner)
        mr = model_runner
        sa = mr.server_args
        # Built first so the cg runners coalesce onto its buffers via the shared
        # input pool; size to the largest tokens/req across modes the worker hits.
        num_tokens_per_bs = 1
        if mr.spec_algorithm.is_speculative():
            # speculative_adaptive can grow draft tokens at runtime; size to the max.
            num_draft_tokens = sa.max_speculative_num_draft_tokens or 1
            if mr.is_draft_worker:
                num_tokens_per_bs = max(
                    sa.speculative_eagle_topk or 1,
                    num_draft_tokens,
                    (
                        2 * (sa.speculative_num_steps or 0)
                        if sa.enable_multi_layer_eagle
                        else 0
                    ),
                )
            else:
                num_tokens_per_bs = (
                    mr.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                        num_draft_tokens, mr.is_draft_worker
                    )
                )
        else:
            dllm_config = DllmConfig.from_server_args(sa)
            if dllm_config is not None:
                # dLLM runs block_size tokens/request (DLLM_EXTEND).
                num_tokens_per_bs = dllm_config.block_size
        max_bs = mr.max_running_requests
        if (
            mr.is_draft_worker
            and mr.spec_algorithm.is_frozen_kv_mtp()
            and sa.speculative_eagle_topk > 1
        ):
            # Frozen-KV MTP expands the draft batch by topk on the bs axis
            # (expand_for_topk_draft) before the eager fallback.
            max_bs *= sa.speculative_eagle_topk
        # Mirror prepare_mlp_sync_batch padding so the registry holds what load_batch copies.
        if require_mlp_sync(sa):
            from sglang.srt.layers.utils.cp_utils import get_cp_padding_align_size

            max_bs = ceil_align(max_bs, self.attn_tp_size)
            max_bs = ceil_align(max_bs, get_cp_padding_align_size())
        prefill_ceiling = (
            sa.max_prefill_buffer_tokens()
            if sa.chunked_prefill_size and sa.chunked_prefill_size > 0
            else mr.max_total_num_tokens
        )
        max_num_token = max(prefill_ceiling, max_bs * num_tokens_per_bs)
        if require_mlp_sync(sa):
            max_num_token = ceil_align(max_num_token, self.attn_tp_size)
            max_num_token = ceil_align(max_num_token, get_cp_padding_align_size())
        self._eager_max_bs = max_bs
        self._eager_num_tokens_per_bs = num_tokens_per_bs
        is_encoder_decoder = mr.model_config.is_encoder_decoder
        self._eager_registry = build_eager_registry(
            device=mr.device,
            max_bs=max_bs,
            max_num_token=max_num_token,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=(
                sa.enable_mamba_extra_buffer() and mr.spec_algorithm.is_none()
            ),
            is_encoder_decoder=is_encoder_decoder,
            encoder_len_fill_value=(
                getattr(mr.model_config.hf_config, "max_source_positions", 0)
                if is_encoder_decoder
                else 0
            ),
            dp_size=sa.dp_size,
        )
        # Eager has no capture step, so warm up here (run-once via mr._kernel_warmed_up).
        self.warmup()

    def _autotune_buffers(self) -> Tuple[Any, int]:
        """Adapter over the eager registry for the autotune dummy forward; fills
        in fields the registry omits (logits buffer, pp_proxy, custom_mask)."""
        mr = self.model_runner
        reg = self._eager_registry
        max_bs = self._eager_max_bs

        def _slot(name):
            return reg.get_slot(name).buffer if reg.has_slot(name) else None

        # num_token_non_padded / global_num_tokens_* are not registered on the
        # eager registry (build_eager_registry passes enable_num_token_non_padded
        # =False, register_global_num_tokens=False); _dummy_run writes + reads
        # them unconditionally, so supply tiny fresh tensors here.
        num_token_non_padded = torch.zeros((1,), dtype=torch.int32, device=mr.device)
        global_dim = (
            mr.server_args.dp_size if require_mlp_tp_gather(mr.server_args) else 1
        )
        global_num_tokens_gpu = torch.zeros(
            (global_dim,), dtype=torch.int32, device=mr.device
        )
        global_num_tokens_for_logprob_gpu = torch.zeros(
            (global_dim,), dtype=torch.int32, device=mr.device
        )

        # custom_mask: only consumed by create_dummy_verify_input (spec). Size it
        # like the decode path's custom_mask for a spec target worker.
        custom_mask: Optional[torch.Tensor] = None
        if mr.spec_algorithm.is_speculative():
            num_tokens_per_bs = self._eager_num_tokens_per_bs
            max_num_token = reg.max_num_tokens
            seq_len_fill_value = mr.attn_backend.get_cuda_graph_seq_len_fill_value()
            custom_mask = torch.ones(
                (max_bs * seq_len_fill_value + max_num_token) * num_tokens_per_bs,
                dtype=torch.bool,
                device=mr.device,
            )

        # pp_proxy_tensors: only read when pp_size>1. _dummy_run slices each value
        # [:pp_hidden_tokens] (pp_hidden_tokens <= num_tokens), so size the first
        # dim to the registry's token ceiling. Mirror _allocate_decode_buffers'
        # keys/dtypes (mHC flattens residual into hidden_states of hc_hidden_size).
        pp_proxy_tensors = None
        if mr.server_args.pp_size > 1:
            hidden_size = mr.model_config.hidden_size
            hc_hidden_size = getattr(mr.model_config, "hc_hidden_size", None)
            is_mhc = hc_hidden_size is not None
            hs = hc_hidden_size if is_mhc else hidden_size
            rows = reg.max_num_tokens
            pp_proxy_tensors = {
                "hidden_states": torch.zeros(
                    (rows, hs), dtype=mr.dtype, device=mr.device
                ),
            }
            if not is_mhc:
                pp_proxy_tensors["residual"] = torch.zeros(
                    (rows, hidden_size), dtype=mr.dtype, device=mr.device
                )

        adapter = SimpleNamespace(
            input_ids=_slot("input_ids"),
            positions=_slot("positions"),
            out_cache_loc=_slot("out_cache_loc"),
            req_pool_indices=_slot("req_pool_indices"),
            seq_lens=_slot("seq_lens"),
            seq_lens_cpu=_slot("seq_lens_cpu"),
            mrope_positions=_slot("mrope_positions"),
            encoder_lens=_slot("encoder_lens"),
            next_token_logits_buffer=None,
            num_token_non_padded=num_token_non_padded,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            custom_mask=custom_mask,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return adapter, max_bs

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        # Eager never runs a cuda graph; callers dispatch on isinstance(...,
        # EagerRunner) and must not route an eager batch into a replay branch.
        return False

    def load_batch(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
    ) -> ForwardBatch:
        """Copy the live batch into the fixed-max eager static buffers (sliced to
        this batch's shape) — the eager counterpart of the cuda-graph runners'
        load_batch."""
        if envs.SGLANG_EAGER_INPUT_NO_COPY.get():
            return replace(forward_batch)
        raw_bs = forward_batch.batch_size
        raw_num_tokens = forward_batch.input_ids.shape[0]
        registry = self._eager_registry
        registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=raw_bs,
            raw_num_tokens=raw_num_tokens,
            padded_num_tokens=raw_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return registry.extract_buffer(
            padded_bs=raw_bs,
            padded_num_tokens=raw_num_tokens,
            forward_batch_template=forward_batch,
        )

    def execute(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None, **kwargs
    ) -> Any:
        mode = forward_batch.forward_mode
        if mode.is_decode():
            return self._execute_decode(forward_batch, pp_proxy_tensors)
        if mode.is_idle():
            return self._execute_idle(forward_batch, pp_proxy_tensors)
        if mode.is_extend(include_draft_extend_v2=True):
            return self._execute_extend(forward_batch, pp_proxy_tensors)
        raise ValueError(f"Invalid forward mode for eager runner: {mode}")

    def _resolve_decode_pdmux(
        self,
    ) -> Tuple[Any, contextlib.AbstractContextManager]:
        """Resolve the (attn_backend, forward_context) the eager decode forward
        runs under. PDmux selects a per-stream backend and publishes it via an
        active ForwardContext; non-pdmux uses attn_backend + the ambient ctx."""
        model_runner = self.model_runner
        if model_runner.server_args.enable_pdmux:
            return model_runner.decode_attn_backend, forward_context(
                ForwardContext(attn_backend=model_runner.decode_attn_backend)
            )
        return model_runner.attn_backend, contextlib.nullcontext()

    def _execute_decode(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        model_runner = self.model_runner
        enable_pdmux = model_runner.server_args.enable_pdmux
        attn_backend, pdmux_ctx = self._resolve_decode_pdmux()
        if not enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)
        if forward_batch.needs_forward_metadata_init():
            if hasattr(model_runner.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                model_runner.model.prepare_forward_batch(forward_batch)
            attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = model_runner._pp_kwargs(pp_proxy_tensors)

        ctx = (
            model_runner.device_timer.wrap(metadata={"category": "decode"})
            if model_runner.device_timer
            else contextlib.nullcontext()
        )

        with ctx, pdmux_ctx:
            return model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

    def _execute_extend(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        model_runner = self.model_runner
        kwargs = model_runner._extend_forward_kwargs(forward_batch, pp_proxy_tensors)

        if not model_runner.server_args.enable_pdmux:
            forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)

        if forward_batch.needs_forward_metadata_init():
            if hasattr(model_runner.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                model_runner.model.prepare_forward_batch(forward_batch)
            model_runner.attn_backend.init_forward_metadata(forward_batch)

        cp_v2_active = is_cp_v2_active(forward_batch)
        forward_positions = forward_batch.positions
        if cp_v2_active:
            prepare_cp_forward(forward_batch)
            complete_hidden_states = kwargs.get("input_embeds")
            if complete_hidden_states is None:
                embed_layer = model_runner.model.get_input_embeddings()
                complete_hidden_states = embed_layer(forward_batch.input_ids)
            sharded_hidden_states, sharded_positions = cp_split_before_forward(
                complete_hidden_states,
                forward_batch.positions,
                forward_batch,
            )
            kwargs["input_embeds"] = sharded_hidden_states
            forward_positions = sharded_positions

        ctx = (
            model_runner.device_timer.wrap(metadata={"category": "extend"})
            if model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            pcg_runner = model_runner.prefill_cuda_graph_runner
            if (
                _is_hip
                and pcg_runner is not None
                and not isinstance(pcg_runner, EagerRunner)
                and not cp_v2_active
            ):
                # HIP PCG eager fallback: enter the PCG context so Dynamo guards
                # and PCG-specific MoE/attention paths stay consistent.
                with (
                    enable_tc_piecewise_cuda_graph(),
                    set_tc_piecewise_forward_context(
                        forward_batch,
                        model_runner.attention_layers,
                        getattr(model_runner.model, "quant_config", None),
                        model_runner.moe_layers,
                        model_runner.moe_fusions,
                        dsa_indexers=model_runner.dsa_indexers,
                    ),
                ):
                    ret = model_runner.model.forward(
                        forward_batch.input_ids,
                        forward_positions,
                        forward_batch,
                        **kwargs,
                    )
            elif cp_v2_active:
                # CP-V2: drive .model directly to gather across CP ranks before logits.
                hidden_states = model_runner.model.model(
                    forward_batch.input_ids,
                    forward_positions,
                    forward_batch,
                    input_embeds=kwargs.get("input_embeds"),
                    pp_proxy_tensors=kwargs.get("pp_proxy_tensors"),
                )
                aux_hidden_states = None
                capture_aux_hidden_states = getattr(
                    model_runner.model, "capture_aux_hidden_states", False
                )
                if capture_aux_hidden_states:
                    hidden_states, aux_hidden_states = hidden_states
                if model_runner.model.pp_group.is_last_rank:
                    hidden_states = cp_gather_after_forward(
                        hidden_states,
                        forward_batch,
                        torch.cuda.current_stream(),
                    )
                    ret = model_runner.model.logits_processor(
                        forward_batch.input_ids,
                        hidden_states,
                        model_runner.model.lm_head,
                        forward_batch,
                        aux_hidden_states,
                    )
                elif capture_aux_hidden_states:
                    ret = hidden_states, aux_hidden_states
                else:
                    ret = hidden_states
            else:
                ret = model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_positions,
                    forward_batch,
                    **kwargs,
                )
        return ret

    def _execute_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        model_runner = self.model_runner
        # Padded idle (DP-attn MLP sync) needs metadata reinit; unpadded must
        # drop stale forward_metadata to avoid an SWA use-after-free on req_pool.
        if forward_batch.batch_size > 0:
            if not model_runner.server_args.enable_pdmux:
                forward_batch = self.load_batch(forward_batch, pp_proxy_tensors)
            model_runner.attn_backend.init_forward_metadata(forward_batch)
        else:
            model_runner.attn_backend.forward_metadata = None

        kwargs = model_runner._pp_kwargs(pp_proxy_tensors)
        ctx = (
            model_runner.device_timer.wrap(metadata={"category": "idle"})
            if model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            return model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
