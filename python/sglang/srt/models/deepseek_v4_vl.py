# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""DeepSeek-V4-Flash-Vision-Exp: vision-enabled wrapper for DeepseekV4.

The HF checkpoint of the vision model reports the same architecture string as
the text-only model ("DeepseekV4ForCausalLM"), so — following the MiMo-V2
precedent — this wrapper takes over the EntryClass registration from
deepseek_v4.py and conditionally builds the vision modules when
``config.vision_n_layers > 0``. Text-only checkpoints behave exactly as
before: the wrapper delegates everything to the text model.

Image-embedding layout (see multimodal/processors/deepseek_v4_vl.py):
each image placeholder expands into a sentinel block of learned vectors
(image_start / image_pad / image_newline / image_end) with ViT+aligner
embeddings scattered into the IMAGE slots in N-layout order (via `perm`).

MoE gate ``bias_vl`` routing (phase 2, done): ``*.gate.bias_vl`` weights
flow through the text model's load_weights onto ``MoEGate.bias_vl``. Hash
layers select image tokens by ``(scores + bias_vl).topk`` instead of the
tid2eid table; non-hash layers fall back to an eager per-token-bias top-k
whenever a batch contains image tokens (see DeepseekV2MoE._forward_topk).

NOT YET IMPLEMENTED (phase 2, required for correct outputs):
- bidirectional / visible-window attention inside image spans during prefill
  (reference: get_image_visible + get_window_topk_idxs_visible); needs
  image-span-aware topk construction in layers/attention/deepseek_v4_backend.py.
"""

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM as _DeepseekV4TextLM
from sglang.srt.models.deepseek_v4_vit import (
    DeepseekV4VisionAligner,
    DeepseekV4VisionTower,
)
from sglang.srt.utils import add_prefix

# Sentinel type ids, matching the reference image_processor.py and
# multimodal/processors/deepseek_v4_vl.py.
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)


class DeepseekV4ForCausalLM(nn.Module):
    """Vision-capable DeepseekV4, registered for arch "DeepseekV4ForCausalLM"."""

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.is_multimodal = getattr(config, "vision_n_layers", 0) > 0

        self.language_model = _DeepseekV4TextLM(
            config, quant_config, prefix=add_prefix("language_model", prefix)
        )

        if self.is_multimodal:
            self.vision = DeepseekV4VisionTower(
                config, prefix=add_prefix("vision", prefix)
            )
            self.aligner = DeepseekV4VisionAligner(
                config, prefix=add_prefix("aligner", prefix)
            )
            self.image_start = nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))
            self.image_pad = nn.Parameter(torch.empty(config.hidden_size))

    # ------------------------------------------------------------------
    # multimodal interface
    # ------------------------------------------------------------------
    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """ViT + aligner + sentinel block assembly, one block per image item.

        Returns [sum(block_len), hidden] concatenated in item order, matching
        the placeholder spans the scheduler padded into input_ids."""
        assert self.is_multimodal
        device = self.image_start.device
        dtype = self.image_start.dtype
        # Order matters: index i of this stack is sentinel type id i; the IMAGE
        # slot (id 2) maps to image_pad and gets overwritten below.
        sentinel = torch.stack(
            [
                self.image_start,
                self.image_pad,
                self.image_pad,
                self.image_newline,
                self.image_end,
            ]
        )
        features = []
        for item in items:
            data = item.model_specific_data
            n_vit_h, n_vit_w = data["n_vit_h"], data["n_vit_w"]
            types = torch.tensor(data["types"], dtype=torch.int64, device=device)
            perm = torch.tensor(data["perm"], dtype=torch.int64, device=device)
            patches = item.feature.to(device=device, dtype=dtype)

            embeds = self.aligner(
                self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w
            )[perm]
            block = sentinel[types]
            image_mask = types == IMAGE
            assert embeds.size(0) == int(image_mask.sum()), (
                f"aligner produced {embeds.size(0)} tokens but the block has "
                f"{int(image_mask.sum())} IMAGE slots"
            )
            block[image_mask] = embeds
            features.append(block)
        return torch.cat(features, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return hidden_states

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------
    _SENTINEL_NAMES = ("image_start", "image_end", "image_newline", "image_pad")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        llm_weights = []
        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.")):
                param = params_dict[name]
                default_weight_loader(param, loaded_weight)
            elif name in self._SENTINEL_NAMES:
                param = params_dict[name]
                default_weight_loader(param, loaded_weight)
            else:
                llm_weights.append((name, loaded_weight))
        self.language_model.load_weights(llm_weights)

    # ------------------------------------------------------------------
    # delegations expected by the framework / engine
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    @property
    def pp_group(self):
        return self.language_model.pp_group

    @property
    def lm_head(self):
        # DSpark's draft worker attaches the target's lm_head to the draft.
        return self.language_model.lm_head

    def set_dspark_layers_to_capture(self, layer_ids) -> None:
        # DSPARK aux-hidden capture is configured on the text model.
        self.language_model.set_dspark_layers_to_capture(layer_ids)


EntryClass = [DeepseekV4ForCausalLM]
