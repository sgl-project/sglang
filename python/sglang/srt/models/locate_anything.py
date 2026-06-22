# SPDX-License-Identifier: Apache-2.0
# Adapted from https://huggingface.co/nvidia/LocateAnything-3B/blob/main/modeling_locateanything.py
# and from vllm-project/vllm PR #44182.
"""Inference-only LocateAnything-3B model for SGLang.

LocateAnything-3B is a multimodal grounding/detection model:

  * MoonViT vision encoder (reused unchanged from Kimi-VL)
  * An InternVL-style ``mlp1`` projector (LayerNorm applied AFTER the 2x2 patch
    merge, i.e. over ``hidden_size * merge_h * merge_w``)
  * A Qwen2 language-model backbone

The model emits structured grounding outputs such as
``<ref>object</ref><box>...</box>`` when special tokens are preserved
(``skip_special_tokens=False``). An optional constrained-decoding logit
processor (:class:`LocateAnythingBoxGrammarLogitProcessor`) restricts the tokens
emitted inside a ``<box>...</box>`` block to a valid ``none`` / point / bbox
pattern.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.kimi_vl_moonvit import MoonViTConfig
from sglang.srt.configs.locate_anything import LocateAnythingConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.kimi_vl_moonvit import MoonVitPretrainedModel
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class LocateAnythingMultiModalProjector(nn.Module):
    """InternVL-style ``mlp1`` projector.

    Unlike Kimi-VL's projector (which LayerNorms the per-patch features over
    ``hidden_size`` *before* the 2x2 merge), LocateAnything merges first and then
    LayerNorms over the merged width ``hidden_size * merge_h * merge_w``.

    HF checkpoint layout (``mlp1`` Sequential):
        mlp1.0 = LayerNorm(merged_size)
        mlp1.1 = Linear(merged_size, text_hidden)
        mlp1.2 = GELU
        mlp1.3 = Linear(text_hidden, text_hidden)
    """

    def __init__(self, config: LocateAnythingConfig):
        super().__init__()

        merge = config.vision_config.merge_kernel_size
        self.merged_size = config.vision_config.hidden_size * merge[0] * merge[1]
        text_hidden = config.text_config.hidden_size

        self.pre_norm = nn.LayerNorm(self.merged_size, eps=1e-5)
        self.linear_1 = nn.Linear(self.merged_size, text_hidden, bias=True)
        # Plain (exact, erf-based) GELU to match the HF checkpoint's nn.GELU().
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden, text_hidden, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # MoonViT's patch_merger yields per-image tensors of shape
        # (num_merged_tokens, merge_h * merge_w, hidden_size); concatenated and
        # flattened to (num_merged_tokens, merged_size) the 4 sub-patches sit
        # contiguously per token, matching the trained LayerNorm(merged_size).
        # reshape (not view) since the concatenated input may be non-contiguous.
        hidden_states = image_features.reshape(-1, self.merged_size)
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LocateAnythingForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: LocateAnythingConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        assert isinstance(config.vision_config, MoonViTConfig)

        self.vision_tower = MoonVitPretrainedModel(config.vision_config)
        self.multi_modal_projector = LocateAnythingMultiModalProjector(config)
        self.quant_config = quant_config

        self.language_model = Qwen2ForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = (
            torch.cat([item.feature for item in items], dim=0)
            .type(self.vision_tower.dtype)
            .to(self.vision_tower.device)
        )

        # Already-projected embeddings (e.g. precomputed) pass through.
        if (
            pixel_values.dim() == 2
            and pixel_values.shape[-1] == self.config.text_config.hidden_size
        ):
            return pixel_values

        # image_grid_hws may arrive as numpy arrays from the HF image processor;
        # coerce each to a tensor before concatenating.
        image_grid_hws = torch.cat(
            [torch.as_tensor(item.image_grid_hws) for item in items], dim=0
        ).to(self.vision_tower.device)
        image_features = self.vision_tower(pixel_values, image_grid_hws)
        assert isinstance(image_features, list)
        return self.multi_modal_projector(torch.cat(image_features))

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        # Remap HF checkpoint prefixes onto SGLang submodule names.
        prefix_mapping = {
            "vision_model.": "vision_tower.",
            "mlp1.0.": "multi_modal_projector.pre_norm.",
            "mlp1.1.": "multi_modal_projector.linear_1.",
            "mlp1.3.": "multi_modal_projector.linear_2.",
        }

        # Qwen2 packs qkv / gate-up; apply the same shard mapping for the LM part.
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        tie_word_embeddings = getattr(
            self.config.text_config, "tie_word_embeddings", False
        )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for src, dst in prefix_mapping.items():
                if name.startswith(src):
                    name = dst + name[len(src) :]
                    break

            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            # Under tied embeddings the checkpoint's lm_head duplicates the input
            # embedding and has no separate destination.
            if tie_word_embeddings and name.startswith("language_model.lm_head."):
                continue

            is_vision_weight = name.startswith("vision_tower.") or name.startswith(
                "multi_modal_projector."
            )

            if is_vision_weight:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue

            # Language-model weights: apply Qwen2 stacked shard mapping.
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        # Reconcile: warn about any model parameter that never received a weight,
        # so a partial/mismatched checkpoint is visible in the logs rather than
        # silently serving garbage. Tied lm_head shares embed_tokens' storage and
        # is loaded via it, so it is expected to be absent here.
        missing = set(params_dict.keys()) - loaded_params
        if tie_word_embeddings:
            missing = {
                n for n in missing if not n.startswith("language_model.lm_head.")
            }
        if missing:
            logger.warning(
                f"LocateAnything: {len(missing)} parameters did not receive "
                f"weights, e.g. {sorted(missing)[:10]}"
            )

        return loaded_params


class LocateAnythingBoxGrammarLogitProcessor(CustomLogitProcessor):
    """Constrained decoding for LocateAnything ``<box>...</box>`` blocks.

    Outside an open box the logits are untouched. Inside an open box (a
    ``box_start`` with no matching ``box_end`` yet) the next token is restricted
    so that the box body is one of:

      * ``none``                       -> ``[none]``
      * a 2-coordinate point           -> ``[c, c]``
      * a 4-coordinate bounding box     -> ``[c, c, c, c]``

    where ``c`` is any token in ``[coord_start_token_id, coord_end_token_id]``.

    Token ids are read per-request from ``custom_param_list[i]`` (keys
    ``box_start_token_id``, ``box_end_token_id``, ``coord_start_token_id``,
    ``coord_end_token_id``, ``none_token_id``) so the processor stays generic.
    The ``__req__`` entry supplies the generated-so-far token ids.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        neg_inf = float("-inf")
        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue
            req = params.get("__req__")
            if req is None:
                continue

            box_start = params.get("box_start_token_id")
            box_end = params.get("box_end_token_id")
            coord_start = params.get("coord_start_token_id")
            coord_end = params.get("coord_end_token_id")
            none_id = params.get("none_token_id")
            if None in (box_start, box_end, coord_start, coord_end, none_id):
                continue

            # Boxes are always generated (never in the prompt), so only the
            # output tokens need scanning — this avoids an O(prompt_len) reverse
            # scan per decode step over the long <IMG_CONTEXT> run.
            output_ids = list(req.output_ids)

            # Find the last box_start; if a box_end follows it, no box is open.
            try:
                last_open = len(output_ids) - 1 - output_ids[::-1].index(box_start)
            except ValueError:
                continue  # no box opened yet
            body = output_ids[last_open + 1 :]
            if box_end in body:
                continue  # last box already closed

            num_coords = sum(1 for t in body if coord_start <= t <= coord_end)
            has_none = none_id in body

            # Determine which token classes are allowed next. The coordinate
            # range is contiguous, so it is masked as a slice rather than an
            # enumerated set (the range can span ~1000 ids per decode step).
            allow_coords = False
            allow_scalars: Set[int] = set()
            if has_none:
                allow_scalars = {box_end}
            elif num_coords == 0:
                allow_coords, allow_scalars = True, {none_id}
            elif num_coords in (1, 3):
                allow_coords = True
            elif num_coords == 2:
                allow_coords, allow_scalars = True, {box_end}
            else:  # >= 4 coords -> must close
                allow_scalars = {box_end}

            mask = torch.full_like(logits[batch_idx], neg_inf)
            if allow_coords:
                mask[coord_start : coord_end + 1] = logits[
                    batch_idx, coord_start : coord_end + 1
                ]
            for tok in allow_scalars:
                mask[tok] = logits[batch_idx, tok]
            logits[batch_idx] = mask

        return logits


EntryClass = [LocateAnythingForConditionalGeneration]
