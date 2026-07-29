# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The SwissAI Initiative
"""Inference-only discrete image/audio fusion for Apertus 1.5."""

import copy
from collections.abc import Iterable
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    Apertus1p5Config,
    Apertus1p5VisionTokenizerModel,
    AutoConfig,
    AutoModel,
)

from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.apertus import ApertusForCausalLM
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import add_prefix, flatten_nested_list


_DEFAULT_IMAGE_TOKEN_OFFSET = 131272
_DEFAULT_AUDIO_TOKEN_OFFSET = 262344


def _init_component_model(
    component_config: Any,
    model_cls: type[nn.Module] | None = None,
) -> nn.Module:
    config_dict = component_config.to_dict()
    config = AutoConfig.for_model(config_dict.pop("model_type"), **config_dict)
    return AutoModel.from_config(config) if model_cls is None else model_cls(config)


class _ApertusLogitsProcessor(LogitsProcessor):
    """Expose impossible input-only Apertus IDs to every logits consumer."""

    def __init__(self, config, input_vocab_size: int) -> None:
        super().__init__(config)
        self.input_vocab_size = input_vocab_size

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
        use_logits_buffer: bool = True,
    ) -> torch.Tensor:
        logits = super()._get_logits(
            hidden_states,
            lm_head,
            logits_metadata,
            embedding_bias=embedding_bias,
            use_logits_buffer=use_logits_buffer,
        )
        if logits.shape[-1] < self.input_vocab_size:
            logits = F.pad(
                logits,
                (0, self.input_vocab_size - logits.shape[-1]),
                value=torch.finfo(logits.dtype).min,
            )
        return logits


class Apertus1p5ForConditionalGeneration(ApertusForCausalLM):
    """Apertus 1.5 with discrete vision and audio codes embedded by Apertus."""

    def __init__(
        self,
        config: Apertus1p5Config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        # The parent owns the Apertus decoder and receives its nested text
        # config; the composite config stays available for the two tokenizers.
        self.mm_config = config
        super().__init__(config.text_config, quant_config=quant_config, prefix=prefix)

        self.vision_tower: nn.Module | None = None
        self.audio_tower: nn.Module | None = None
        if self.pp_group.is_first_rank:
            # Both tokenizers assign discrete codes by nearest-neighbour/argmax.
            # Keeping them in fp32 is required for checkpoint parity.
            with set_default_torch_dtype(torch.float32):
                self.vision_tower = _init_component_model(
                    config.vision_tokenizer_config,
                    model_cls=Apertus1p5VisionTokenizerModel,
                )
                self.audio_tower = _init_component_model(config.audio_tokenizer_config)

        self.image_token_offset = getattr(
            config, "image_token_offset", _DEFAULT_IMAGE_TOKEN_OFFSET
        )
        self.audio_token_offset = getattr(
            config, "audio_token_offset", _DEFAULT_AUDIO_TOKEN_OFFSET
        )
        self._input_vocab_size = config.text_config.vocab_size
        self._output_vocab_size = getattr(
            config.text_config,
            "output_vocab_size",
            self._input_vocab_size,
        )
        self._pad_logits_to_input_vocab = (
            self._output_vocab_size != self._input_vocab_size
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

        if self._pad_logits_to_input_vocab:
            if self.config.tie_word_embeddings:
                raise ValueError(
                    "Apertus 1.5 cannot tie input embeddings when output_vocab_size "
                    "is smaller than the multimodal input vocabulary."
                )
            if self.pp_group.is_last_rank:
                self.lm_head = ParallelLMHead(
                    self._output_vocab_size,
                    self.config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_server_args().enable_dp_lm_head,
                )
            logits_config = copy.copy(self.config)
            logits_config.vocab_size = self._output_vocab_size
            self.logits_processor = _ApertusLogitsProcessor(
                logits_config, self._input_vocab_size
            )

        if not self.pp_group.is_last_rank:
            self.lm_head = PPMissingLayer()

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
        parameter = next(module.parameters())
        return parameter.device, parameter.dtype

    def _embed_code_ids(self, ids_per_item: List[torch.Tensor]) -> torch.Tensor:
        if not ids_per_item:
            return torch.empty(
                0,
                self.config.hidden_size,
                device=self.model.embed_tokens.weight.device,
                dtype=self.model.embed_tokens.weight.dtype,
            )
        all_ids = torch.cat(ids_per_item).to(self.model.embed_tokens.weight.device)
        embeddings = self.model.embed_tokens(all_ids)

        return embeddings

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        assert self.vision_tower is not None
        device, dtype = self._module_device_dtype(self.vision_tower)
        image_ids = []
        for image in flatten_nested_list([item.feature for item in items]):
            if not isinstance(image, torch.Tensor):
                continue
            if image.dim() == 3:
                image = image.unsqueeze(0)
            with torch.inference_mode():
                codes = self.vision_tower.encode(image.to(device=device, dtype=dtype))
            image_ids.append(
                codes.flatten().to(torch.long) + self.image_token_offset
            )
        return self._embed_code_ids(image_ids)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        assert self.audio_tower is not None
        device, dtype = self._module_device_dtype(self.audio_tower)
        audio_ids = []
        for audio in flatten_nested_list([item.feature for item in items]):
            if not isinstance(audio, torch.Tensor):
                continue
            with torch.inference_mode():
                output = self.audio_tower.encode(
                    audio.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
                )
            audio_ids.append(
                output.audio_codes.squeeze(0).squeeze(0).to(torch.long)
                + self.audio_token_offset
            )
        return self._embed_code_ids(audio_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        component_tensors = {}
        if self.pp_group.is_first_rank:
            for component_name, component in (
                ("vision_tower", self.vision_tower),
                ("audio_tower", self.audio_tower),
            ):
                if component is not None:
                    component_tensors.update(
                        {
                            f"{component_name}.{name}": tensor
                            for name, tensor in component.named_parameters()
                        }
                    )
                    component_tensors.update(
                        {
                            f"{component_name}.{name}": tensor
                            for name, tensor in component.named_buffers()
                        }
                    )

        def remapped_weights():
            for name, weight in weights:
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model.") :]
                elif name.startswith("model.vision_tokenizer."):
                    name = "vision_tower." + name[len("model.vision_tokenizer.") :]
                elif name.startswith("model.audio_tokenizer."):
                    name = "audio_tower." + name[len("model.audio_tokenizer.") :]

                yield name, weight

        def load_component_weight(name: str, weight: torch.Tensor) -> bool:
            if not name.startswith(("vision_tower.", "audio_tower.")):
                return False

            if not self.pp_group.is_first_rank:
                return True

            component_tensor = component_tensors.get(name)
            if component_tensor is None:
                raise ValueError(
                    f"No vision/audio tensor matches checkpoint key: {name}"
                )
            weight_loader = getattr(
                component_tensor, "weight_loader", default_weight_loader
            )
            weight_loader(component_tensor, weight)
            return True

        def filter_language_weights():
            for name, weight in remapped_weights():
                if not load_component_weight(name, weight):
                    yield name, weight

        return super().load_weights(filter_language_weights())


EntryClass = [Apertus1p5ForConditionalGeneration]
