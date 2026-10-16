# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3.5 inference for complete Valen + Mixer Hugging Face weights."""

from typing import Iterable

import torch

from sglang.srt.layers.valen_mixer import ValenMixerHead
from sglang.srt.layers.valen_pooler import ValenMixerPooler
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration


class ValenQwenForDecisionMaking(Qwen3_5ForConditionalGeneration):
    def __init__(self, config, quant_config=None, prefix=""):
        if quant_config is not None:
            raise ValueError("Valen quantized serving has not been validated")
        if (
            config.qwen_execution != "shared_state"
            or config.valen_head.get("head_type") != "mixer"
        ):
            raise ValueError("Valen requires shared_state execution with a Mixer head")
        if not config.tie_word_embeddings:
            raise ValueError(
                "Valen serving requires the released tied-embedding Qwen3.5 backbone"
            )
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        settings = {
            key: value for key, value in config.valen_head.items() if key != "head_type"
        }
        self.pooler = ValenMixerPooler(
            ValenMixerHead(config.text_config.hidden_size, **settings)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        head_weights = {}

        def backbone_weights():
            for name, tensor in weights:
                if name.startswith("head."):
                    head_weights[name.removeprefix("head.")] = tensor
                elif name.startswith("backbone.language_model."):
                    # Use the conditional model's path, not its bare-backbone
                    # path (which expects an unrelated LM decision readout).
                    yield (
                        "model.language_model."
                        + name.removeprefix("backbone.language_model."),
                        tensor,
                    )
                elif name.startswith("backbone.visual."):
                    yield (
                        "model.visual." + name.removeprefix("backbone.visual."),
                        tensor,
                    )
                else:
                    raise ValueError(f"Unexpected Valen checkpoint tensor: {name}")

        loaded = super().load_weights(backbone_weights())
        self.pooler.head.load_state_dict(head_weights, strict=True)
        loaded.update(
            f"pooler.head.{name}" for name, _ in self.pooler.head.named_parameters()
        )
        expected = {
            name
            for name, _ in self.named_parameters()
            if not name.startswith("lm_head.")
        }
        missing = expected - loaded
        if missing:
            raise ValueError(f"Incomplete Valen backbone load: {sorted(missing)}")
        return loaded

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        if not get_embedding:
            raise ValueError(
                "Valen returns decisions through /v1/valen or /v1/systemone"
            )
        return super().forward(
            input_ids,
            positions,
            forward_batch,
            get_embedding=True,
            pp_proxy_tensors=pp_proxy_tensors,
        )


EntryClass = ValenQwenForDecisionMaking
