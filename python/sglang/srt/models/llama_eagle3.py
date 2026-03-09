"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.utils import add_prefix

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

import copy
import itertools
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

logger = logging.getLogger(__name__)

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = LlamaMLP(
            config.hidden_size, inter_size, config.hidden_act, quant_config, prefix
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.is_mrope_enabled = (
            hasattr(config, "rope_scaling")
            and config.rope_scaling is not None
            and "mrope_section" in config.rope_scaling
        )
        # fix rope_scaling for qwen2.5-vl
        if self.is_mrope_enabled:
            config.rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        self.midlayer = LlamaDecoderLayer(config, 0, quant_config, prefix)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Multi-expert state (populated by register_experts)
        self._has_experts = False
        self._num_experts = 0
        self._profile_names: List[str] = []
        self._default_profile_idx: int = 0
        self.fc_experts: Optional[nn.ModuleList] = None
        self.midlayer_experts: Optional[nn.ModuleList] = None
        self.norm_experts: Optional[nn.ModuleList] = None

    def register_experts(
        self,
        profile_names: List[str],
        default_profile_idx: int,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        """Create N expert copies of fc, midlayer, norm for MoE-style parallel routing.

        Each expert midlayer uses a distinct ``layer_id`` (0, 1, …) so that
        it has its own independent KV cache layer.  The worker must call
        ``extend_layers(N - 1)`` on the KV pool **before** CUDA graph
        capture to allocate the additional layers.

        In ``_forward_moe`` each expert processes **only** its own subset
        of tokens through FC → midlayer (attention + MLP) → norm, which
        is correct because:

        * Attention metadata (``kv_indptr`` / ``kv_indices``) is rebuilt
          on-the-fly for the subset.
        * Each expert's KV cache layer is isolated, so writes from
          different experts never collide.
        """
        n = len(profile_names)
        if n <= 1:
            return

        self._has_experts = True
        self._num_experts = n
        self._profile_names = profile_names
        self._default_profile_idx = default_profile_idx

        fc_list: List[nn.Module] = []
        mid_list: List[nn.Module] = []
        norm_list: List[nn.Module] = []

        for i in range(n):
            if i == default_profile_idx:
                # Reuse the existing modules (layer_id=0)
                fc_list.append(self.fc)
                mid_list.append(self.midlayer)
                norm_list.append(self.norm)
            else:
                # Create new instances with unique layer_id for independent KV cache
                new_fc = torch.nn.Linear(
                    self.hidden_size_in * 3,
                    config.hidden_size,
                    bias=getattr(config, "bias", False),
                ).to(
                    device=next(self.fc.parameters()).device,
                    dtype=next(self.fc.parameters()).dtype,
                )

                new_mid = LlamaDecoderLayer(
                    config, layer_id=i, quant_config=quant_config, prefix=prefix
                ).to(
                    device=next(self.midlayer.parameters()).device,
                    dtype=next(self.midlayer.parameters()).dtype,
                )

                new_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
                    device=next(self.norm.parameters()).device,
                    dtype=next(self.norm.parameters()).dtype,
                )

                fc_list.append(new_fc)
                mid_list.append(new_mid)
                norm_list.append(new_norm)

        self.fc_experts = nn.ModuleList(fc_list)
        self.midlayer_experts = nn.ModuleList(mid_list)
        self.norm_experts = nn.ModuleList(norm_list)

        logger.info(
            f"Registered {n} MoE experts (independent KV cache layers) "
            f"for profiles: {profile_names}, default_idx={default_profile_idx}"
        )

    def _get_profile_ids(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
        """Extract per-token profile IDs from forward_batch.

        In EAGLE3 draft decode, each token in the batch corresponds to a request.
        The profile is determined by the request's custom_params['eagle_draft_profile'].

        Returns:
            A tensor of shape [num_tokens] with integer profile indices,
            or None if multi-expert is not active.
        """
        if not self._has_experts:
            return None

        custom_params_list = None
        if (
            forward_batch.sampling_info is not None
            and forward_batch.sampling_info.custom_params is not None
        ):
            custom_params_list = forward_batch.sampling_info.custom_params

        num_tokens = forward_batch.input_ids.shape[0]

        if custom_params_list is None:
            # All tokens use default profile
            return torch.full(
                (num_tokens,),
                self._default_profile_idx,
                dtype=torch.long,
                device=forward_batch.input_ids.device,
            )

        # Build per-request profile index
        # In draft decode, batch_size = num_reqs, and num_tokens = batch_size * topk
        # custom_params_list has one entry per request
        num_reqs = len(custom_params_list)
        profile_name_to_idx = {name: i for i, name in enumerate(self._profile_names)}

        req_profile_ids = []
        for cp in custom_params_list:
            if isinstance(cp, dict):
                p = cp.get("eagle_draft_profile")
                if p is not None and p in profile_name_to_idx:
                    req_profile_ids.append(profile_name_to_idx[p])
                else:
                    req_profile_ids.append(self._default_profile_idx)
            else:
                req_profile_ids.append(self._default_profile_idx)

        req_profile_tensor = torch.tensor(
            req_profile_ids, dtype=torch.long, device=forward_batch.input_ids.device
        )

        # Expand from per-request to per-token
        if num_tokens == num_reqs:
            # 1:1 mapping (e.g. draft extend or prefill)
            return req_profile_tensor
        elif num_tokens > num_reqs and num_tokens % num_reqs == 0:
            # topk tokens per request
            topk = num_tokens // num_reqs
            return req_profile_tensor.repeat_interleave(topk)
        else:
            # Fallback: use default for all
            return torch.full(
                (num_tokens,),
                self._default_profile_idx,
                dtype=torch.long,
                device=forward_batch.input_ids.device,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = forward_batch.spec_info.hidden_states

        # idle batch
        if hidden_states.shape[0] == 0:
            if hidden_states.shape[-1] != embeds.shape[-1]:
                hidden_states = self.fc(hidden_states)
            return hidden_states, [hidden_states]

        # ----- Multi-expert MoE-style parallel forward -----
        if self._has_experts:
            # CUDA graph capture cannot handle dynamic ops like torch.unique()
            # in _forward_moe(). During capture we always use the default expert
            # so the captured graph records a fixed computation path.
            # At replay time, the worker layer detects mixed-profile batches
            # and falls back to eager mode where _forward_moe() runs normally.
            if torch.cuda.is_current_stream_capturing():
                return self._forward_single_expert(
                    self._default_profile_idx,
                    positions,
                    embeds,
                    hidden_states,
                    forward_batch,
                )
            return self._forward_moe(
                input_ids, positions, forward_batch, embeds, hidden_states
            )

        # ----- Single-profile (original) forward -----
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]

    @staticmethod
    def _can_extract_sub_metadata(forward_metadata) -> bool:
        """Check whether the attention metadata supports sub-batch extraction.

        Currently only the Triton backend's ``ForwardMetadata`` (which exposes
        ``kv_indptr`` / ``kv_indices`` as plain tensors) can be efficiently
        sliced.  Other backends (FlashInfer, FA3, …) store their indexing
        state inside opaque wrapper objects, so we must fall back to the
        full-batch-per-expert path for those.
        """
        return (
            forward_metadata is not None
            and hasattr(forward_metadata, "kv_indptr")
            and hasattr(forward_metadata, "kv_indices")
        )

    @staticmethod
    def _build_expert_sub_metadatas(
        full_metadata,
        expert_indices: "List[Optional[torch.Tensor]]",
        expert_indices_cpu: "List[Optional[List[int]]]",
        device: torch.device,
    ) -> "List[Optional[object]]":
        """Pre-compute sub-batch ``ForwardMetadata`` for *all* experts.

        Designed to minimise GPU→CPU synchronisation:

        * A single ``kv_indptr.tolist()`` transfers the full CSR to CPU.
        * Per-expert CSR construction uses the CPU-side index lists
          (``expert_indices_cpu``) directly — **zero** per-expert GPU→CPU
          transfers.
        * ``kv_indices`` gathering uses one ``torch.cat`` of GPU slices
          per expert (no CPU readback of indices).

        Args:
            expert_indices: GPU tensors of token indices per expert.
            expert_indices_cpu: Parallel CPU lists (avoids x_idx.tolist()).
        """
        from sglang.srt.layers.attention.triton_backend import ForwardMetadata

        full_kv_indptr = full_metadata.kv_indptr  # [full_bs + 1]
        full_kv_indices = full_metadata.kv_indices  # [total_kv_entries]

        # Single bulk GPU→CPU transfer (the only sync point)
        full_kv_indptr_cpu = full_kv_indptr.tolist()

        results: List[Optional[object]] = []
        for x_idx, x_idx_cpu in zip(expert_indices, expert_indices_cpu):
            if x_idx is None:
                results.append(None)
                continue

            n_sub = len(x_idx_cpu)

            # Build CSR boundaries from the CPU copies (zero GPU ops)
            starts_cpu = [full_kv_indptr_cpu[i] for i in x_idx_cpu]
            ends_cpu = [full_kv_indptr_cpu[i + 1] for i in x_idx_cpu]
            lengths_cpu = [e - s for s, e in zip(starts_cpu, ends_cpu)]

            # Upload compact CSR to GPU in one shot
            cumsum = list(itertools.accumulate(lengths_cpu, initial=0))
            sub_kv_indptr = torch.tensor(
                cumsum, dtype=full_kv_indptr.dtype, device=device
            )

            # Gather kv_indices: Python-level slicing of GPU tensor
            total = cumsum[-1]
            if total > 0:
                sub_kv_indices = torch.cat(
                    [full_kv_indices[s:e] for s, e in zip(starts_cpu, ends_cpu)]
                )
            else:
                sub_kv_indices = torch.empty(
                    0, dtype=full_kv_indices.dtype, device=device
                )

            sub_attn_logits = (
                full_metadata.attn_logits[:n_sub]
                if full_metadata.attn_logits is not None
                else None
            )
            sub_attn_lse = (
                full_metadata.attn_lse[:n_sub]
                if full_metadata.attn_lse is not None
                else None
            )
            sub_num_kv_splits = (
                full_metadata.num_kv_splits[x_idx]
                if full_metadata.num_kv_splits is not None
                else None
            )

            results.append(
                ForwardMetadata(
                    attn_logits=sub_attn_logits,
                    attn_lse=sub_attn_lse,
                    max_extend_len=full_metadata.max_extend_len,
                    num_kv_splits=sub_num_kv_splits,
                    kv_indptr=sub_kv_indptr,
                    kv_indices=sub_kv_indices,
                    qo_indptr=full_metadata.qo_indptr,
                    custom_mask=full_metadata.custom_mask,
                    mask_indptr=full_metadata.mask_indptr,
                    window_kv_indptr=full_metadata.window_kv_indptr,
                    window_kv_indices=full_metadata.window_kv_indices,
                    window_num_kv_splits=full_metadata.window_num_kv_splits,
                    window_kv_offsets=full_metadata.window_kv_offsets,
                )
            )

        return results

    def _forward_moe(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """MoE-style forward: route tokens to per-expert sub-networks.

        Two execution modes are supported depending on the attention backend:

        **Sub-batch mode** (Triton backend – preferred):
          Each expert processes *only* its own tokens through
          FC → midlayer (attention + MLP) → norm.  We temporarily replace
          ``forward_batch.attn_backend.forward_metadata`` with a sub-batch
          version built from the full-batch CSR (``kv_indptr``/``kv_indices``).
          Total work is proportional to 1× batch size.

          All sub-batch metadatas are pre-built *before* the expert loop
          via ``_build_expert_sub_metadatas`` so that the single GPU→CPU
          transfer (``kv_indptr.tolist()``) is amortised across experts.

        **Full-batch mode** (FlashInfer / other backends – fallback):
          These backends store attention metadata inside opaque wrappers that
          cannot be sliced.  Each expert therefore runs the midlayer on the
          **full batch** (to keep kv_indptr/kv_indices consistent), and we
          only scatter the expert's own tokens from the output.  FC and norm
          are still computed on the expert's subset only.
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device

        # ----- Build per-expert token indices entirely on CPU -----
        # This avoids all GPU→CPU synchronisations (unique, mask.any,
        # nonzero) that would otherwise stall the pipeline.
        expert_indices_cpu: List[Optional[List[int]]] = [
            None for _ in range(self._num_experts)
        ]
        active_count = 0
        single_expert_idx = -1

        custom_params_list = None
        if (
            forward_batch.sampling_info is not None
            and forward_batch.sampling_info.custom_params is not None
        ):
            custom_params_list = forward_batch.sampling_info.custom_params

        if custom_params_list is not None:
            num_reqs = len(custom_params_list)
            profile_name_to_idx = {
                name: i for i, name in enumerate(self._profile_names)
            }

            # Determine per-request expert index (CPU only)
            req_experts: List[int] = []
            for cp in custom_params_list:
                if isinstance(cp, dict):
                    p = cp.get("eagle_draft_profile")
                    if p is not None and p in profile_name_to_idx:
                        req_experts.append(profile_name_to_idx[p])
                    else:
                        req_experts.append(self._default_profile_idx)
                else:
                    req_experts.append(self._default_profile_idx)

            # Expand per-request → per-token indices
            if num_tokens > num_reqs and num_tokens % num_reqs == 0:
                topk = num_tokens // num_reqs
            elif num_tokens == num_reqs:
                topk = 1
            else:
                topk = 1

            # Bucket token indices by expert (pure Python, zero GPU ops)
            buckets: List[List[int]] = [[] for _ in range(self._num_experts)]
            for req_idx, eidx in enumerate(req_experts):
                base = req_idx * topk
                for t in range(topk):
                    buckets[eidx].append(base + t)

            for eidx in range(self._num_experts):
                if buckets[eidx]:
                    expert_indices_cpu[eidx] = buckets[eidx]
                    active_count += 1
                    single_expert_idx = eidx
        else:
            # No custom_params → all tokens use default expert
            expert_indices_cpu[self._default_profile_idx] = list(range(num_tokens))
            active_count = 1
            single_expert_idx = self._default_profile_idx

        # Fast-path: all tokens share one profile
        if active_count <= 1:
            return self._forward_single_expert(
                (
                    single_expert_idx
                    if single_expert_idx >= 0
                    else self._default_profile_idx
                ),
                positions,
                embeds,
                hidden_states,
                forward_batch,
            )

        # Upload indices to GPU (one tensor per expert, single sync point)
        expert_indices: List[Optional[torch.Tensor]] = []
        for cpu_idx in expert_indices_cpu:
            if cpu_idx is not None:
                expert_indices.append(
                    torch.tensor(cpu_idx, dtype=torch.long, device=device)
                )
            else:
                expert_indices.append(None)

        need_fc = hidden_states.shape[-1] != embeds.shape[-1]
        hidden_dim = self.config.hidden_size

        # Allocate output buffers
        hs_logits_out = torch.empty(
            num_tokens, hidden_dim, device=device, dtype=embeds.dtype
        )
        hs_aux_out = torch.empty(
            num_tokens, hidden_dim, device=device, dtype=embeds.dtype
        )

        # Determine execution mode and pre-build sub-batch metadatas
        attn_backend = forward_batch.attn_backend
        orig_forward_metadata = (
            attn_backend.forward_metadata if attn_backend is not None else None
        )
        use_sub_batch = self._can_extract_sub_metadata(orig_forward_metadata)

        # Pre-build all expert sub-metadatas OUTSIDE the loop so that the
        # GPU→CPU transfer (kv_indptr.tolist()) happens once in bulk.
        sub_metadatas: List[Optional[object]] = [None] * self._num_experts
        if use_sub_batch:
            sub_metadatas = self._build_expert_sub_metadatas(
                orig_forward_metadata, expert_indices, expert_indices_cpu, device
            )

        # Save originals that we will temporarily mutate
        orig_out_cache_loc = forward_batch.out_cache_loc

        for expert_idx in range(self._num_experts):
            x_idx = expert_indices[expert_idx]
            if x_idx is None:
                continue

            if use_sub_batch:
                # ---- Sub-batch mode (Triton) ----
                if need_fc:
                    expert_hs = self.fc_experts[expert_idx](hidden_states[x_idx])
                else:
                    expert_hs = hidden_states[x_idx]

                expert_embeds = embeds[x_idx]
                expert_positions = positions[x_idx]

                if orig_out_cache_loc is not None:
                    forward_batch.out_cache_loc = orig_out_cache_loc[x_idx]

                # Swap in the pre-computed sub-batch metadata
                attn_backend.forward_metadata = sub_metadatas[expert_idx]

                residual = None
                hs_expert_out, residual = self.midlayer_experts[expert_idx](
                    expert_positions,
                    expert_embeds,
                    expert_hs,
                    forward_batch,
                    residual,
                )

                hs_to_logits, hs_to_aux = self.norm_experts[expert_idx](
                    hs_expert_out, residual
                )

                hs_logits_out[x_idx] = hs_to_logits
                hs_aux_out[x_idx] = hs_to_aux
            else:
                # ---- Full-batch mode (FlashInfer / others) ----
                if need_fc:
                    expert_hs = self.fc_experts[expert_idx](hidden_states[x_idx])
                    hs_full = torch.zeros(
                        num_tokens, hidden_dim, device=device, dtype=embeds.dtype
                    )
                    hs_full[x_idx] = expert_hs
                else:
                    hs_full = hidden_states

                # Midlayer: full batch (attention metadata is opaque)
                residual = None
                hs_expert, residual = self.midlayer_experts[expert_idx](
                    positions,
                    embeds,
                    hs_full,
                    forward_batch,
                    residual,
                )

                # Norm + scatter only this expert's tokens
                hs_to_logits, hs_to_aux = self.norm_experts[expert_idx](
                    hs_expert, residual
                )
                hs_logits_out[x_idx] = hs_to_logits[x_idx]
                hs_aux_out[x_idx] = hs_to_aux[x_idx]

        # Restore originals
        forward_batch.out_cache_loc = orig_out_cache_loc
        if attn_backend is not None:
            attn_backend.forward_metadata = orig_forward_metadata

        return hs_logits_out, [hs_aux_out]

    def _forward_single_expert(
        self,
        expert_idx: int,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Fast path: all tokens use the same expert."""
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc_experts[expert_idx](hidden_states)

        residual = None
        hidden_states, residual = self.midlayer_experts[expert_idx](
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hs_to_logits, hs_to_aux = self.norm_experts[expert_idx](hidden_states, residual)

        return hs_to_logits, [hs_to_aux]


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        # EAGLE3 draft model has exactly 1 hidden layer (midlayer).
        # Multi-expert KV cache layers are extended dynamically after pool creation.
        if self.config.num_hidden_layers != 1:
            raise ValueError(
                f"EAGLE3 requires exactly 1 hidden layer, got {self.config.num_hidden_layers}"
            )

        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = (
            config_.draft_vocab_size
        )  # draft logits processor has it's own vocab size
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

        # Multi-expert state
        self._has_experts = False
        self._num_experts = 0
        self._profile_names: List[str] = []
        self._default_profile_idx: int = 0
        self.lm_head_experts: Optional[nn.ModuleList] = None

    # ------------------------------------------------------------------ #
    #  Multi-expert registration
    # ------------------------------------------------------------------ #

    def register_experts(
        self,
        profile_names: List[str],
        default_profile_idx: int,
    ):
        """Register multiple lm_head experts for MoE-style routing.

        Also delegates to self.model.register_experts() for fc/midlayer/norm.
        The lm_head experts are created here since lm_head is on the CausalLM level.

        The default expert reuses self.lm_head.
        """
        n = len(profile_names)
        if n <= 1:
            return

        self._has_experts = True
        self._num_experts = n
        self._profile_names = profile_names
        self._default_profile_idx = default_profile_idx

        # Create lm_head experts
        lm_head_list: List[nn.Module] = []
        for i in range(n):
            if i == default_profile_idx:
                lm_head_list.append(self.lm_head)
            else:
                if self.config.tie_word_embeddings:
                    lm_head_list.append(self.model.embed_tokens)
                else:
                    new_lm_head = ParallelLMHead(
                        self.config.draft_vocab_size,
                        self.config.hidden_size,
                        quant_config=self.quant_config,
                    ).to(
                        device=next(self.lm_head.parameters()).device,
                        dtype=next(self.lm_head.parameters()).dtype,
                    )
                    lm_head_list.append(new_lm_head)

        self.lm_head_experts = nn.ModuleList(lm_head_list)

        # Register experts in the inner model (fc, midlayer, norm)
        self.model.register_experts(
            profile_names=profile_names,
            default_profile_idx=default_profile_idx,
            config=self.config,
            quant_config=self.quant_config,
        )

        logger.info(f"LlamaForCausalLMEagle3 registered {n} experts: {profile_names}")

    def _extract_own_profile_weights(self) -> Dict[str, torch.Tensor]:
        """Extract (clone) the current model weights that are profile-specific."""
        weights: Dict[str, torch.Tensor] = {}
        for name, param in self.model.fc.named_parameters():
            weights[f"fc.{name}"] = param.data.clone()
        for name, param in self.model.midlayer.named_parameters():
            weights[f"midlayer.{name}"] = param.data.clone()
        for name, param in self.model.norm.named_parameters():
            weights[f"norm.{name}"] = param.data.clone()
        for name, param in self.lm_head.named_parameters():
            weights[f"lm_head.{name}"] = param.data.clone()
        return weights

    def load_expert_weights(
        self,
        expert_idx: int,
        weights: Dict[str, torch.Tensor],
    ):
        """Load weights into a specific expert (fc, midlayer, norm, lm_head).

        Args:
            expert_idx: which expert to load into
            weights: dict of param_name -> tensor (keys like fc.weight, midlayer.*, etc.)
        """
        if not self._has_experts:
            raise RuntimeError("Experts not registered yet")

        # Load fc expert weights
        for name, param in self.model.fc_experts[expert_idx].named_parameters():
            key = f"fc.{name}"
            if key in weights:
                param.data.copy_(weights[key])

        # Load midlayer expert weights (need to handle different layer_id)
        for name, param in self.model.midlayer_experts[expert_idx].named_parameters():
            key = f"midlayer.{name}"
            if key in weights:
                param.data.copy_(weights[key])

        # Load norm expert weights
        for name, param in self.model.norm_experts[expert_idx].named_parameters():
            key = f"norm.{name}"
            if key in weights:
                param.data.copy_(weights[key])

        # Load lm_head expert weights
        if self.lm_head_experts is not None:
            for name, param in self.lm_head_experts[expert_idx].named_parameters():
                key = f"lm_head.{name}"
                if key in weights:
                    param.data.copy_(weights[key])

    # ------------------------------------------------------------------ #
    #  Override forward for MoE routing
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional["PPProxyTensors"] = None,
    ):
        if not self._has_experts:
            # Original single-profile forward
            return super().forward(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        # MoE-style forward with multiple experts
        # self.model.forward() handles fc/midlayer/norm routing internally
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                # During CUDA graph capture, _moe_logits() calls
                # profile_ids.unique() which is not supported.
                # Use the default expert lm_head for a fixed compute path.
                if torch.cuda.is_current_stream_capturing():
                    return self.logits_processor(
                        input_ids,
                        hidden_states,
                        self.lm_head_experts[self.model._default_profile_idx],
                        forward_batch,
                        aux_hidden_states,
                    )
                # MoE logits: route to different lm_heads per token
                return self._moe_logits(
                    input_ids, hidden_states, forward_batch, aux_hidden_states
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    def _moe_logits(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        aux_hidden_states: Optional[List[torch.Tensor]],
    ) -> LogitsProcessorOutput:
        """Compute logits using per-token expert lm_heads.

        Uses CPU-side routing (same logic as ``_forward_moe``) to avoid
        GPU→CPU synchronisations in the hot path.
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device

        # ---- CPU-side routing (zero GPU sync) ----
        custom_params_list = None
        if (
            forward_batch.sampling_info is not None
            and forward_batch.sampling_info.custom_params is not None
        ):
            custom_params_list = forward_batch.sampling_info.custom_params

        expert_buckets: List[Optional[List[int]]] = [
            None for _ in range(self._num_experts)
        ]
        active_count = 0
        single_expert_idx = self._default_profile_idx

        if custom_params_list is not None:
            num_reqs = len(custom_params_list)
            profile_name_to_idx = {
                name: i for i, name in enumerate(self.model._profile_names)
            }
            if num_tokens > num_reqs and num_tokens % num_reqs == 0:
                topk = num_tokens // num_reqs
            else:
                topk = 1

            buckets: List[List[int]] = [[] for _ in range(self._num_experts)]
            for req_idx, cp in enumerate(custom_params_list):
                eidx = self._default_profile_idx
                if isinstance(cp, dict):
                    p = cp.get("eagle_draft_profile")
                    if p is not None and p in profile_name_to_idx:
                        eidx = profile_name_to_idx[p]
                base = req_idx * topk
                for t in range(topk):
                    buckets[eidx].append(base + t)

            for eidx in range(self._num_experts):
                if buckets[eidx]:
                    expert_buckets[eidx] = buckets[eidx]
                    active_count += 1
                    single_expert_idx = eidx
        else:
            expert_buckets[self._default_profile_idx] = list(range(num_tokens))
            active_count = 1

        # Fast path: single expert
        if active_count <= 1:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head_experts[single_expert_idx],
                forward_batch,
                aux_hidden_states,
            )

        # --- Mixed profiles ---
        # Step 1: full logits_processor with default lm_head
        result = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head_experts[self._default_profile_idx],
            forward_batch,
            aux_hidden_states,
        )

        # Step 2: for each non-default expert, compute lm_head on its tokens
        for expert_idx in range(self._num_experts):
            if expert_idx == self._default_profile_idx:
                continue

            cpu_idx = expert_buckets[expert_idx]
            if cpu_idx is None:
                continue

            x_idx = torch.tensor(cpu_idx, dtype=torch.long, device=device)

            expert_hidden = hidden_states[x_idx]
            lm_head = self.lm_head_experts[expert_idx]

            if hasattr(lm_head, "weight"):
                expert_logits = torch.matmul(
                    expert_hidden.to(lm_head.weight.dtype), lm_head.weight.T
                )
            else:
                expert_logits = lm_head.quant_method.apply(lm_head, expert_hidden, None)

            result.next_token_logits[x_idx] = expert_logits.to(
                result.next_token_logits.dtype
            )

        return result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def get_hot_token_id(self):
        return self.hot_token_id


EntryClass = [LlamaForCausalLMEagle3]
