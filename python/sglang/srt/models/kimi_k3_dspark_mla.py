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
"""Kimi-K3 DSpark draft model with native MLA attention."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDecoderLayer, DFlashDraftModel
from sglang.srt.models.dspark import (
    DSparkDraftMixin,
    _DSPARK_SKIPPED_WEIGHT_PREFIXES,
)
from sglang.srt.models.kimi_k3 import KimiK3MLAAttention
from sglang.srt.runtime_context import get_device, get_parallel
from sglang.srt.utils.common import BumpAllocator

logger = logging.getLogger(__name__)


class KimiDSparkMLAAttention(KimiK3MLAAttention):
    """K3 MLA attention adapted to the draft checkpoint's RoPE and mask."""

    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_id,
            quant_config=quant_config,
            all_reduce_fusion=False,
            prefix="self_attn",
        )

        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters:
            rope_theta = float(rope_parameters.get("rope_theta", 10000.0))
            rope_scaling = (
                rope_parameters
                if rope_parameters.get("rope_type", "default") != "default"
                else None
            )
        else:
            rope_theta = float(getattr(config, "rope_theta", 10000.0))
            rope_scaling = getattr(config, "rope_scaling", None)

        # The target K3 model uses NoPE MLA and therefore constructs its
        # attention with skip_rope=True. The draft checkpoint was trained with
        # interleaved RoPE for both proposal queries and context keys.
        self.context_rotary_emb = get_rope_wrapper(
            int(config.qk_rope_head_dim),
            rotary_dim=int(config.qk_rope_head_dim),
            max_position=int(config.max_position_embeddings),
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=not getattr(config, "rope_interleave", True),
            device=get_device().device,
        )
        self.rotary_emb = self.context_rotary_emb

        # Draft attention is non-causal inside the proposal block. Speculative
        # metadata supplies the exact dual-source visibility mask.
        self.attn_mqa.attn_type = AttentionType.ENCODER_ONLY
        self.attn_mha.attn_type = AttentionType.ENCODER_ONLY

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # The target runs with DCP, while this compact draft remains ordinarily
        # TP-sharded. Keep target-only DCP/LSE behavior out of draft attention.
        with get_parallel().override(
            dcp_enabled=False,
            dcp_size=1,
            dcp_rank=0,
            attn_dcp_size=1,
            attn_dcp_rank=0,
        ):
            get_attn_tp_context().set_attn_inputs(
                AttentionInputs(hidden_states, forward_batch, self.prepare_qkv_latent)
            )
            zero_allocator = BumpAllocator(
                buffer_size=2,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            output = super().forward(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )
        return output[0] if isinstance(output, tuple) else output


class KimiDSparkMLADecoderLayer(DFlashDecoderLayer):
    attention_cls = KimiDSparkMLAAttention


class KimiK3DSparkMLADraftModel(DSparkDraftMixin, DFlashDraftModel):
    """Dense DSpark backbone whose draft KV cache uses Kimi-K3 MLA latents."""

    decoder_layer_cls = KimiDSparkMLADecoderLayer
    supports_fused_context_kv = False

    @staticmethod
    def _resolve_param_name(name: str, params_dict: dict) -> Optional[str]:
        if name in params_dict:
            return name
        if name.startswith("model."):
            stripped = name[len("model.") :]
            return stripped if stripped in params_dict else None
        prefixed = f"model.{name}"
        return prefixed if prefixed in params_dict else None

    def _load_backbone_weights(
        self, weights: list[Tuple[str, torch.Tensor]], params_dict: dict
    ) -> None:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        cached_a_proj: dict[str, torch.Tensor] = {}

        for name, loaded_weight in weights:
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                resolved = self._resolve_param_name(mapped, params_dict)
                if resolved is None:
                    continue
                param = params_dict[resolved]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight, shard_id)
                handled = True
                break
            if handled:
                continue

            if (
                ".self_attn.q_a_proj." in name
                or ".self_attn.kv_a_proj_with_mqa." in name
            ):
                cached_a_proj[name] = loaded_weight
                q_name = (
                    name
                    if ".q_a_proj." in name
                    else name.replace(".kv_a_proj_with_mqa.", ".q_a_proj.")
                )
                kv_name = (
                    name
                    if ".kv_a_proj_with_mqa." in name
                    else name.replace(".q_a_proj.", ".kv_a_proj_with_mqa.")
                )
                if q_name in cached_a_proj and kv_name in cached_a_proj:
                    fused = torch.cat(
                        [cached_a_proj[q_name], cached_a_proj[kv_name]], dim=0
                    )
                    fused_name = q_name.replace(
                        ".q_a_proj.", ".fused_qkv_a_proj_with_mqa."
                    )
                    resolved = self._resolve_param_name(fused_name, params_dict)
                    if resolved is None:
                        raise ValueError(
                            f"Kimi DSpark MLA fused projection {fused_name!r} "
                            "is missing"
                        )
                    param = params_dict[resolved]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, fused)
                    cached_a_proj.pop(q_name)
                    cached_a_proj.pop(kv_name)
                continue

            resolved = self._resolve_param_name(name, params_dict)
            if resolved is None:
                logger.warning("Kimi DSpark MLA: skipping unexpected weight %s", name)
                continue
            param = params_dict[resolved]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)

        if cached_a_proj:
            raise ValueError(
                "Kimi DSpark MLA checkpoint has incomplete q_a/kv_a projection "
                f"pairs: {sorted(cached_a_proj)}"
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        markov_weights = []
        confidence_weights = []
        backbone_weights = []
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if any(name.startswith(p) for p in _DSPARK_SKIPPED_WEIGHT_PREFIXES):
                continue
            if name.startswith("confidence_head."):
                if self.confidence_head is not None:
                    confidence_weights.append((name, loaded_weight))
            elif name.startswith("markov_head."):
                markov_weights.append((name, loaded_weight))
            else:
                backbone_weights.append((name, loaded_weight))

        self._load_backbone_weights(backbone_weights, params_dict)
        for name, loaded_weight in markov_weights:
            if name not in params_dict:
                raise ValueError(f"Kimi DSpark MLA unexpected Markov weight {name!r}")
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
        self._load_confidence_weights(
            confidence_weights=confidence_weights, params_dict=params_dict
        )
        self.post_load_weights()

    def post_load_weights(self) -> None:
        """Prepare the absorbed MLA key/value weights used by decode kernels."""
        for layer in self.layers:
            attn = layer.self_attn
            weight = attn.kv_b_proj.weight
            if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise NotImplementedError(
                    "Kimi DSpark MLA serving currently requires float kv_b weights; "
                    f"got {weight.dtype}"
                )
            w_kc, w_vc = weight.unflatten(
                0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)
            ).split([attn.qk_nope_head_dim, attn.v_head_dim], dim=1)
            attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            attn.w_vc = w_vc.contiguous().transpose(1, 2)

    @staticmethod
    def _context_latent(
        attn: KimiDSparkMLAAttention, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_weight = attn.fused_qkv_a_proj_with_mqa.weight
        kv_weight = fused_weight[attn.q_lora_rank :]
        latent = F.linear(hidden, kv_weight)
        k_nope, k_rope = latent.split(
            [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
        )
        return attn.kv_a_layernorm(k_nope), k_rope

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Project target hidden states into the draft model's MLA KV cache."""
        ctx_hidden = self.project_target_hidden(target_hidden)
        loc = cache_loc
        if cache_loc_2d is not None and commit_lens is not None:
            width = cache_loc_2d.shape[1]
            valid = (
                torch.arange(width, device=cache_loc_2d.device).unsqueeze(0)
                < commit_lens.to(torch.long).unsqueeze(1)
            ).reshape(-1)
            loc = cache_loc_2d.reshape(-1)[valid]
            ctx_hidden = ctx_hidden[valid]
            positions = positions[valid]
        if loc.numel() == 0:
            return

        for layer in self.layers:
            attn = layer.self_attn
            k_nope, k_rope = self._context_latent(attn, ctx_hidden)
            k_rope = k_rope.unsqueeze(1)
            dummy_q = torch.empty_like(k_rope)
            _, k_rope = attn.context_rotary_emb(positions, dummy_q, k_rope)
            cache_k = torch.cat((k_nope.unsqueeze(1), k_rope), dim=-1)
            pool.set_kv_buffer(
                attn.attn_mqa,
                loc,
                cache_k,
                k_nope.unsqueeze(1),
                attn.attn_mqa.k_scale,
                attn.attn_mqa.v_scale,
            )


EntryClass = [KimiK3DSparkMLADraftModel]
