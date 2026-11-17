# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Nemotron Decision: a bidirectional Ministral3 with a vocabulary readout."""

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.ministral3 import Ministral3ForCausalLM
from sglang.srt.models.utils import WeightsMapper


class NemotronForDecision(Ministral3ForCausalLM):
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_prefix={"encoder.": "model.", "diffusion_head.": "lm_head."}
    )

    def __init__(self, config, quant_config=None, prefix=""):
        if config.is_causal or config.sliding_window is not None:
            raise ValueError("Nemotron Decision requires full bidirectional attention")
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        for layer in self.model.layers[self.model.start_layer : self.model.end_layer]:
            layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY
            # Query scaling uses the training window, not the extended context limit.
            layer.self_attn.max_position_embeddings = config.rope_parameters.get(
                "original_max_position_embeddings", config.max_position_embeddings
            )

    def load_weights(self, weights):
        return super().load_weights(self.hf_to_sglang_mapper.apply(weights))


EntryClass = NemotronForDecision
